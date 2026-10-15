"""slides: outline, then per-slide content, visuals and speaker script from the approved Brief."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .brief import verify_approval
from .chunk import estimate_tokens
from .hashing import hash_file, hash_obj
from .io_utils import read_json, write_json, write_text
from .llm import SPLITTABLE, FatalLLMError, SchemaValidationError, TruncatedError, describe_error
from .prompts import (OUTLINE_SCHEMA, PROMPT_VERSIONS, SLIDE_CONTENT_SCHEMA, SLIDE_SCHEMA,
                      SLIDE_SCRIPT_SCHEMA, outline_instructions, slide_instructions)
from .state import Context, StageStore, RunReport, model_usage

log = logging.getLogger("course_kb")
OUTLINE_PROCESSOR = "outline-1"
SLIDE_PROCESSOR = "slide-1"


def _fp(ctx: Context, prompt: str, schemas: list[dict]) -> dict:
    s = ctx.settings
    return {"prompt": PROMPT_VERSIONS[prompt], "schemas": [hash_obj(x) for x in schemas],
            "model": s.text_model, "base_url": s.base_url, "lang": s.lang}


# ---------------------------------------------------------------- outline

def _outline_card(c: dict) -> dict:
    return {"card_id": c["card_id"], "concept": c["concept"], "definition": c["definition"][:300],
            "prerequisites": c["prerequisites"], "example_count": len(c["examples"]),
            "has_formula": bool(c["formulas"])}


def _outline_call(ctx: Context, header: dict, chapter: dict, cards: list[dict], required: list[str]) -> list[dict]:
    allowed = [c["card_id"] for c in cards]
    payload = {**header, "chapter": {k: chapter[k] for k in ("title", "learning_objectives")},
               "required_card_ids": [c for c in required if c in allowed],
               "cards": [_outline_card(c) for c in cards]}
    feedback = None
    for _ in range(2):
        if feedback:
            payload["validation_feedback"] = feedback
        data, _meta = ctx.llm.structured("text", "slide_outline", OUTLINE_SCHEMA,
                                         outline_instructions(ctx.settings.lang),
                                         json.dumps(payload, ensure_ascii=False), ctx.settings.output_tokens)
        bad = sorted({cid for s in data["slides"] for cid in s["card_ids"] if cid not in allowed})
        empty = [s["title"] for s in data["slides"] if not s["card_ids"]]
        if not bad and not empty:
            return data["slides"]
        feedback = f"unknown card_ids={bad}; slides without card_ids={empty}"
    raise SchemaValidationError(f"outline invalid after retry: {feedback}")


def _outline_chapter(ctx: Context, header: dict, chapter: dict, cards: list[dict], required: list[str]) -> list[dict]:
    """Plan a chapter in consecutive batches that respect the input budget; on truncation or
    HTTP 413 halve the batch (bounded by the number of cards)."""
    budget = ctx.settings.outline_batch_tokens
    batches: list[list[dict]] = []
    cur, tokens = [], 0
    for c in cards:
        t = estimate_tokens(json.dumps(_outline_card(c), ensure_ascii=False))
        if cur and tokens + t > budget:
            batches.append(cur)
            cur, tokens = [], 0
        cur.append(c)
        tokens += t
    if cur:
        batches.append(cur)

    def run(batch: list[dict]) -> list[dict]:
        try:
            return _outline_call(ctx, header, chapter, batch, required)
        except SPLITTABLE:
            if len(batch) <= 1:
                raise
            mid = len(batch) // 2
            log.warning("outline batch of %d cards too large; splitting", len(batch))
            return run(batch[:mid]) + run(batch[mid:])

    return [s for b in batches for s in run(b)]


def _fix_coverage(slides: list[dict], chapter_cards: list[str], required: list[str], by_id: dict) -> list[dict]:
    """Insert a slide for every required card the model left out, at its order position."""
    pos = {c: i for i, c in enumerate(chapter_cards)}
    covered = {c for s in slides for c in s["card_ids"]}
    for cid in [c for c in required if c not in covered]:
        new = {"title": by_id[cid]["concept"], "learning_objective": f"說明{by_id[cid]['concept']}",
               "card_ids": [cid], "added_by": "coverage_check"}
        idx = next((i for i, s in enumerate(slides) if min(pos[c] for c in s["card_ids"]) > pos[cid]),
                   len(slides))
        slides.insert(idx, new)
    return slides


def build_outline(ctx: Context, bound: dict, binding: dict, store: StageStore, report: RunReport,
                  force: bool) -> dict:
    brief, cards = bound["brief"], bound["cards"]
    by_id = {c["card_id"]: c for c in cards}
    excluded = {x["card_id"] for x in brief.get("excluded_slide_cards", [])}
    header = {"course_title": brief.get("title"), "audience": brief.get("audience"),
              "course_objectives": brief.get("objectives", []), "language": ctx.settings.lang}
    chapters = brief["chapters"]
    report.start_stage("slides_outline", [ch["chapter_id"] for ch in chapters])
    out_slides, failed = [], []
    for ch in chapters:
        chid = ch["chapter_id"]
        ch_cards = [by_id[c] for c in ch["card_ids"]]
        required = [c for c in ch["card_ids"] if c not in excluded]
        part = ctx.ws.p("slides", "outline_parts", f"{chid}.json")
        content = hash_obj({"binding": binding, "header": header, "chapter": ch,
                            "cards": [by_id[c] for c in ch["card_ids"]], "required": required})
        fp = {"content": content, "processor": OUTLINE_PROCESSOR, "budget": ctx.settings.outline_batch_tokens,
              **_fp(ctx, "outline", [OUTLINE_SCHEMA])}
        key = hash_obj(fp)
        if not required:
            slides = []
            report.set_unit("slides_outline", chid, "success", "no required cards", content)
        elif not force and store.is_valid("slides_outline", chid, key, [part]):
            slides = read_json(part)["slides"]
            report.set_unit("slides_outline", chid, "skipped", "cache_hit", content)
        else:
            try:
                with model_usage() as used:
                    slides = _outline_chapter(ctx, header, ch, ch_cards, required)
                slides = _fix_coverage(slides, ch["card_ids"], required, by_id)
                write_json(part, {"chapter_id": chid, "slides": slides})
                store.record_success("slides_outline", chid, key, content, [part], fingerprint=fp, models=used)
                report.set_unit("slides_outline", chid, "success", f"{len(slides)} slides", content)
            except Exception as e:  # noqa: BLE001
                store.record_failure("slides_outline", chid, key, content, describe_error(e))
                report.set_unit("slides_outline", chid, "failed", getattr(e, "kind", type(e).__name__),
                                content, describe_error(e))
                failed.append(chid)
                log.error("outline %s failed: %s", chid, e)
                if isinstance(e, FatalLLMError):
                    raise
                continue
        for n, s in enumerate(slides, 1):
            out_slides.append({"slide_id": f"{chid}-s{n:02d}", "chapter_id": chid, **s})
    report.finish_stage("slides_outline")
    outline = {**binding, "failed_chapters": failed, "slides": out_slides}
    write_json(ctx.ws.outline_file, outline)
    return outline


def outline_order_errors(slides: list[dict], order: dict) -> list[str]:
    first = {}
    for i, s in enumerate(slides):
        for c in s["card_ids"]:
            first.setdefault(c, i)
    errors = []
    for e in order["effective_edges"]:
        a, b = e["from"], e["to"]
        if a in first and b in first and first[a] > first[b]:
            errors.append(f"prerequisite {a} first appears after {b}")
    return errors


# ---------------------------------------------------------------- per slide

def _slide_inputs(slide: dict, by_id: dict) -> tuple[dict, dict, dict]:
    ev_alias, frame_alias = {}, {}
    cards_payload = []
    for cid in slide["card_ids"]:
        c = by_id[cid]
        refs = []
        for e in c["evidence"]:
            alias = next((a for a, x in ev_alias.items() if x["segment_id"] == e["segment_id"]), None)
            if alias is None:
                alias = f"E{len(ev_alias) + 1}"
                ev_alias[alias] = e
            refs.append(alias)
        for v in c["visual_refs"]:
            if not any(x.get("frame_id") == v.get("frame_id") for x in frame_alias.values()):
                frame_alias[f"F{len(frame_alias) + 1}"] = v
        cards_payload.append({
            "card_id": cid, "concept": c["concept"], "definition": c["definition"],
            "intuition": c["intuition"], "examples": [x["text"] for x in c["examples"]],
            "formulas": [x["latex"] for x in c["formulas"]],
            "misconceptions": [x["text"] for x in c["misconceptions"]],
            "conflicts": [{"topic": k["topic"], "claims": [cl["statement"] for cl in k["claims"]]}
                          for k in c["conflicts"]],
            "evidence_refs": refs})
    payload = {"cards": cards_payload,
               "evidence": [{"ref": a, "source_id": e["source_id"], "locator": e["locator"], "text": e["text"]}
                            for a, e in ev_alias.items()],
               "frames": [{"ref": a, "time": v.get("start"), "description": v.get("description")}
                          for a, v in frame_alias.items()]}
    return payload, ev_alias, frame_alias


def _check_refs(data: dict, ev_alias: dict, frame_alias: dict) -> None:
    if not data["evidence_refs"]:
        raise SchemaValidationError("slide has no evidence_refs")
    bad = [r for r in data["evidence_refs"] if r not in ev_alias]
    if bad:
        raise SchemaValidationError(f"unknown evidence_refs {bad}")
    for v in data.get("visuals", []):
        if v["frame_ref"] and v["frame_ref"] not in frame_alias:
            raise SchemaValidationError(f"unknown frame_ref {v['frame_ref']}")


def generate_slide(ctx: Context, header: dict, slide: dict, neighbours: dict, by_id: dict) -> dict:
    payload, ev_alias, frame_alias = _slide_inputs(slide, by_id)
    payload = {**header, **neighbours,
               "slide": {k: slide[k] for k in ("title", "learning_objective")}, **payload}
    lang = ctx.settings.lang
    text = json.dumps(payload, ensure_ascii=False)
    try:
        data, _ = ctx.llm.structured("text", "slide_full", SLIDE_SCHEMA, slide_instructions(lang),
                                     text, ctx.settings.output_tokens)
        _check_refs(data, ev_alias, frame_alias)
        mode = "full"
    except TruncatedError:
        log.warning("slide %s truncated; splitting content and script", slide["slide_id"])
        part1, _ = ctx.llm.structured("text", "slide_content", SLIDE_CONTENT_SCHEMA,
                                      slide_instructions(lang, "content"), text, ctx.settings.output_tokens)
        _check_refs(part1, ev_alias, frame_alias)
        payload2 = {**payload, "final_content": part1["content"],
                    "final_visuals": [v["description"] for v in part1["visuals"]]}
        part2, _ = ctx.llm.structured("text", "slide_script", SLIDE_SCRIPT_SCHEMA,
                                      slide_instructions(lang, "script"),
                                      json.dumps(payload2, ensure_ascii=False), ctx.settings.output_tokens)
        _check_refs(part2, ev_alias, frame_alias)
        data = {"content": part1["content"], "visuals": part1["visuals"],
                "speaker_script": part2["speaker_script"],
                "evidence_refs": list(dict.fromkeys(part1["evidence_refs"] + part2["evidence_refs"])),
                "review_items": part1["review_items"] + part2["review_items"]}
        mode = "split"
    if not data["speaker_script"].strip() or not data["content"]:
        raise SchemaValidationError("slide content or speaker_script is empty")
    evidence = [ev_alias[r] for r in dict.fromkeys(data["evidence_refs"])]
    visuals = [{"kind": v["kind"], "description": v["description"],
                "frame": frame_alias.get(v["frame_ref"]) if v["frame_ref"] else None}
               for v in data["visuals"]]
    review = [{"type": "model", "text": t} for t in data["review_items"]]
    for e in ev_alias.values():
        if e["kind"] == "frame" and "待確認：" in e["text"]:
            review.append({"type": "frame_uncertain", "frame": e["locator"],
                           "text": e["text"].split("待確認：", 1)[1]})
    for cid in slide["card_ids"]:
        c = by_id[cid]
        for k in c["conflicts"]:
            claims = "；".join(f"{cl['statement']}（{', '.join(cl['card_ids'])}）" for cl in k["claims"])
            review.append({"type": "conflict", "card_id": cid, "text": f"{k['topic']}：{claims}"})
        if c["merge_status"] not in ("single", "merged"):
            review.append({"type": "merge", "card_id": cid, "text": f"merge status: {c['merge_status']}"})
    return {"slide_id": slide["slide_id"], "chapter_id": slide["chapter_id"], "title": slide["title"],
            "learning_objective": slide["learning_objective"], "content": data["content"],
            "visuals": visuals, "speaker_script": data["speaker_script"], "card_ids": slide["card_ids"],
            "evidence_segment_ids": [e["segment_id"] for e in evidence],
            "sources": [{"source_id": e["source_id"], "locator": e["locator"]} for e in evidence],
            "visual_refs": list(frame_alias.values()),
            "review_items": review, "generation_mode": mode}


# ---------------------------------------------------------------- export

def _source_locations(ctx: Context) -> dict:
    m = read_json(ctx.ws.sources_manifest) or {}
    return {s["id"]: s for s in m.get("sources", [])}


def run_slides(ctx: Context, force: bool = False) -> dict:
    bound = verify_approval(ctx)  # stops before any model request when approval drifted
    approval = bound["approval"]
    binding = {k: approval[k] for k in ("cards_version", "brief_hash", "order_hash", "completeness_hash")}
    brief, cards, order = bound["brief"], bound["cards"], bound["order"]
    by_id = {c["card_id"]: c for c in cards}
    store, report = ctx.stages(), ctx.report()
    outline = build_outline(ctx, bound, binding, store, report, force)
    slides = outline["slides"]
    chapters = {ch["chapter_id"]: ch for ch in brief["chapters"]}
    header = {"course_title": brief.get("title"), "audience": brief.get("audience"),
              "language": ctx.settings.lang}
    report.start_stage("slides", [s["slide_id"] for s in slides])
    store.drop_missing("slides", [s["slide_id"] for s in slides])

    def do(i: int) -> Optional[dict]:
        s = slides[i]
        ch = chapters[s["chapter_id"]]
        neighbours = {
            "chapter": {"title": ch["title"], "learning_objectives": ch.get("learning_objectives", [])},
            "previous_slide": {"title": slides[i - 1]["title"]} if i > 0 else None,
            "next_slide": {"title": slides[i + 1]["title"]} if i + 1 < len(slides) else None,
        }
        out = ctx.ws.slide_items / f"{s['slide_id']}.json"
        content = hash_obj({"binding": binding, "slide": s, "neighbours": neighbours, "header": header,
                            "cards": [by_id[c] for c in s["card_ids"]]})
        fp = {"content": content, "processor": SLIDE_PROCESSOR,
              **_fp(ctx, "slide", [SLIDE_SCHEMA, SLIDE_CONTENT_SCHEMA, SLIDE_SCRIPT_SCHEMA])}
        key = hash_obj(fp)
        if not force and store.is_valid("slides", s["slide_id"], key, [out]):
            item = read_json(out)
            if all(item.get(k) == v for k, v in binding.items()):
                report.set_unit("slides", s["slide_id"], "skipped", "cache_hit", content)
                return item
        try:
            with model_usage() as used:
                item = {**binding, **generate_slide(ctx, header, s, neighbours, by_id)}
            write_json(out, item)
            store.record_success("slides", s["slide_id"], key, content, [out], fingerprint=fp, models=used)
            report.set_unit("slides", s["slide_id"], "success", item["generation_mode"], content)
            return item
        except Exception as e:  # noqa: BLE001
            store.record_failure("slides", s["slide_id"], key, content, describe_error(e))
            report.set_unit("slides", s["slide_id"], "failed", getattr(e, "kind", type(e).__name__),
                            content, describe_error(e))
            log.error("slide %s failed: %s", s["slide_id"], e)
            if isinstance(e, FatalLLMError):
                raise
            return None

    with ThreadPoolExecutor(max_workers=max(1, ctx.settings.max_workers)) as pool:
        items = list(pool.map(do, range(len(slides))))
    report.finish_stage("slides")
    return export(ctx, brief, cards, order, binding, outline, items, bound["failures"], approval)


def export(ctx: Context, brief: dict, cards: list[dict], order: dict, binding: dict, outline: dict,
           items: list[Optional[dict]], upstream_failures: list[dict], approval: dict) -> dict:
    slides = outline["slides"]
    by_id = {c["card_id"]: c for c in cards}
    errors = []
    ids = [s["slide_id"] for s in slides]
    if len(ids) != len(set(ids)):
        errors.append("duplicate slide_id")
    chapter_ids = [ch["chapter_id"] for ch in brief["chapters"]]
    if [c for c in dict.fromkeys(s["chapter_id"] for s in slides)] != [c for c in chapter_ids
                                                                       if any(s["chapter_id"] == c for s in slides)]:
        errors.append("slides do not follow the chapter order")
    errors += outline_order_errors(slides, order)
    good = [it for it in items if it]
    for it in good:
        if any(it.get(k) != v for k, v in binding.items()):
            errors.append(f"{it['slide_id']}: bound to a different approval")
        for cid in it["card_ids"]:
            if cid not in by_id:
                errors.append(f"{it['slide_id']}: unknown card {cid}")
        allowed = {e for cid in it["card_ids"] if cid in by_id for e in by_id[cid]["evidence_segment_ids"]}
        if not set(it["evidence_segment_ids"]) <= allowed:
            errors.append(f"{it['slide_id']}: evidence outside its cards")
        for v in it["visuals"]:
            fr = v.get("frame")
            if not fr:
                continue
            path = ctx.ws.root / fr["path"]
            if not path.exists() or hash_file(path) != fr.get("image_hash"):
                errors.append(f"{it['slide_id']}: frame {fr.get('frame_id')} missing or changed on disk")
    covered = {c for it in good for c in it["card_ids"]}
    excluded = {x["card_id"]: x["reason"] for x in brief.get("excluded_slide_cards", [])}
    uncovered = [c["card_id"] for c in cards if c["card_id"] not in covered]
    required_uncovered = [c for c in uncovered if c not in excluded]
    failed_slides = [s["slide_id"] for s, it in zip(slides, items) if it is None]
    complete = not (errors or failed_slides or required_uncovered or upstream_failures
                    or outline["failed_chapters"])
    locs = _source_locations(ctx)
    for it in good:
        for src in it["sources"]:
            meta = locs.get(src["source_id"], {})
            src["location"] = meta.get("location")
            src["title"] = meta.get("title")
    doc = {
        **binding, "status": "complete" if complete else "partial",
        "upstream_partial": bool(upstream_failures),
        "missing": [{"item_id": f["item_id"], "stage": f["stage"], "unit": f["unit"], "reason": f["reason"]}
                    for f in upstream_failures],
        "failed_slides": failed_slides, "failed_outline_chapters": outline["failed_chapters"],
        "validation_errors": errors, "uncovered_card_ids": uncovered,
        "required_uncovered_card_ids": required_uncovered,
        "excluded_slide_cards": brief.get("excluded_slide_cards", []),
        "slides": good,
    }
    write_json(ctx.ws.slides_json, doc)
    write_text(ctx.ws.slides_md, render_slides_md(brief, doc, slides, items, by_id))
    log.info("slides: %d/%d generated (%s)", len(good), len(slides), doc["status"])
    return doc


def render_slides_md(brief: dict, doc: dict, slides: list[dict], items: list, by_id: dict) -> str:
    L = [f"# 投影片規劃：{brief.get('title', '')}", "",
         f"- 狀態：{doc['status']}", f"- cards_version：`{doc['cards_version']}`", ""]
    if doc["missing"]:
        L += ["## 上游缺漏（已人工接受，仍屬 partial）", ""]
        L += [f"- `{m['item_id']}` {m['stage']} / {m['unit']}（{m['reason']}）" for m in doc["missing"]] + [""]
    problems = ([f"驗證錯誤：{e}" for e in doc["validation_errors"]]
                + [f"生成失敗：{s}" for s in doc["failed_slides"]]
                + [f"大綱失敗章節：{c}" for c in doc["failed_outline_chapters"]]
                + [f"未涵蓋的必要卡片：{c}" for c in doc["required_uncovered_card_ids"]])
    if problems:
        L += ["## 待處理", ""] + [f"- {p}" for p in problems] + [""]
    if doc["excluded_slide_cards"]:
        L += ["## 核准排除的卡片", ""]
        L += [f"- {x['card_id']} {by_id.get(x['card_id'], {}).get('concept', '')}：{x['reason']}"
              for x in doc["excluded_slide_cards"]] + [""]
    current = None
    for s, it in zip(slides, items):
        if s["chapter_id"] != current:
            current = s["chapter_id"]
            ch = next(c for c in brief["chapters"] if c["chapter_id"] == current)
            L += [f"## {current} {ch['title']}", ""]
        L += [f"### {s['slide_id']} {s['title']}", "", f"**學習目標**：{s['learning_objective']}", ""]
        if it is None:
            L += ["> ⚠ 此張生成失敗，請重跑 slides 補做。", ""]
            continue
        L += ["**畫面內容**", ""] + [f"- {c}" for c in it["content"]] + [""]
        if it["visuals"]:
            L += ["**視覺安排**", ""]
            for v in it["visuals"]:
                fr = v.get("frame")
                L.append(f"- [{v['kind']}] {v['description']}"
                         + (f"（影格 {fr.get('path')} @ {fr.get('start')}）" if fr else ""))
            L.append("")
        L += ["**講者講稿**", "", it["speaker_script"], ""]
        L.append(f"**卡片**：{', '.join(it['card_ids'])}")
        L.append("")
        L += ["**來源**", ""] + [f"- {src['source_id']}（{src.get('location') or ''}）{src['locator']}"
                                 for src in it["sources"]] + [""]
        if it["review_items"]:
            L += ["**待確認**", ""] + [f"- [{r['type']}] {r['text']}" for r in it["review_items"]] + [""]
    return "\n".join(L)
