"""brief: chapter planning, validation, review decisions, rendering and approval."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from .chunk import estimate_tokens
from .freshness import upstream_problems
from .hashing import hash_obj
from .io_utils import read_json, write_json, write_text
from .llm import SPLITTABLE
from .ordering import compute_order, compute_prereqs, order_hash
from .prompts import PLAN_SCHEMA, PROMPT_VERSIONS, plan_instructions
from .state import Context, RunReport, model_usage, summarize_models
from .cards import cards_version as compute_cards_version

log = logging.getLogger("course_kb")
PLAN_PROCESSOR = "plan-1"
DECISIONS = {
    "map_prerequisite": "prereq",   # requires target_card_id
    "audience_has": "prereq",
    "need_material": "prereq",
    "remove_edge": "edge",
    "accept_missing": "failure",
    "acknowledge": "any",
}


class BriefError(Exception):
    """Blocking problem (stale inputs, missing approval, invalid master file)."""


# ---------------------------------------------------------------- loading

def load_cards(ctx: Context) -> tuple[list[dict], str]:
    data = read_json(ctx.ws.cards_file)
    if not data:
        raise BriefError("no cards/cards.json; run 'cards' first")
    actual = compute_cards_version(data["cards"])
    if actual != data.get("cards_version"):
        raise BriefError("cards/cards.json content does not match its cards_version")
    return data["cards"], actual


def load_prereqs(ctx: Context, cards: list[dict], version: str, allow_compute: bool) -> dict:
    p = read_json(ctx.ws.prereqs_file)
    s = ctx.settings
    fp = {"cards_version": version, "threshold": s.prereq_similarity, "embed_model": s.embed_model}
    if p and p.get("fingerprint") == fp:
        return p
    if not allow_compute:
        raise BriefError("brief/prereqs.json does not match the current cards; run 'brief' to re-plan")
    p = {"fingerprint": fp, **compute_prereqs(cards, ctx.embedder, s.prereq_similarity)}
    write_json(ctx.ws.prereqs_file, p)
    return p


# ---------------------------------------------------------------- planning

def _summary(card: dict) -> dict:
    return {"card_id": card["card_id"], "concept": card["concept"],
            "definition": card["definition"][:200], "prerequisites": card["prerequisites"],
            "example_count": len(card["examples"]), "has_formula": bool(card["formulas"]),
            "conflicts": [c["topic"] for c in card["conflicts"]]}


def _plan_batch(ctx: Context, header: dict, summaries: list[dict], prev_titles: list[str]) -> list[dict]:
    payload = {**header, "previous_chapter_titles": prev_titles, "cards": summaries}
    expected = [c["card_id"] for c in summaries]
    feedback = None
    for attempt in range(2):
        if feedback:
            payload["validation_feedback"] = feedback
        data, _ = ctx.llm.structured("text", "plan_chapters", PLAN_SCHEMA,
                                     plan_instructions(ctx.settings.lang),
                                     json.dumps(payload, ensure_ascii=False), ctx.settings.output_tokens)
        got = [cid for ch in data["chapters"] for cid in ch["card_ids"]]
        problems = []
        if sorted(got) != sorted(expected):
            missing = [c for c in expected if c not in got]
            extra = [c for c in got if c not in expected or got.count(c) > 1]
            problems.append(f"missing={missing} unknown_or_duplicate={sorted(set(extra))}")
        elif got != expected:
            problems.append("card order must follow the given order")
        if not problems:
            return data["chapters"]
        feedback = "; ".join(problems)
        log.warning("plan batch invalid (attempt %d): %s", attempt + 1, feedback)
    return data["chapters"]  # kept as a draft; validation reports the errors


def plan_chapters(ctx: Context, header: dict, ordered: list[dict]) -> list[dict]:
    budget = ctx.settings.brief_batch_tokens
    summaries = [_summary(c) for c in ordered]

    def run(batch: list[dict], prev: list[str]) -> list[dict]:
        try:
            return _plan_batch(ctx, header, batch, prev)
        except SPLITTABLE:
            if len(batch) <= 1:
                raise
            mid = len(batch) // 2
            log.warning("plan batch of %d cards too large; splitting", len(batch))
            first = run(batch[:mid], prev)
            return first + run(batch[mid:], prev + [c["title"] for c in first])

    chapters: list[dict] = []
    batch: list[dict] = []
    tokens = 0
    for s in summaries:
        t = estimate_tokens(json.dumps(s, ensure_ascii=False))
        if batch and tokens + t > budget:
            chapters += run(batch, [c["title"] for c in chapters])
            batch, tokens = [], 0
        batch.append(s)
        tokens += t
    if batch:
        chapters += run(batch, [c["title"] for c in chapters])
    return [{"chapter_id": f"ch{i:02d}", **ch} for i, ch in enumerate(chapters, 1)]


def _plan_fingerprint(ctx: Context, version: str, prereqs: dict, header: dict) -> dict:
    s = ctx.settings
    return {"cards_version": version, "prereqs": hash_obj(prereqs), "header": header,
            "prompt": PROMPT_VERSIONS["plan"], "schema": hash_obj(PLAN_SCHEMA),
            "model": s.text_model, "base_url": s.base_url, "lang": s.lang,
            "budget": s.brief_batch_tokens, "processor": PLAN_PROCESSOR}


def run_brief(ctx: Context, title: Optional[str] = None, audience: Optional[str] = None,
              objectives: Optional[list[str]] = None, force: bool = False) -> dict:
    ws = ctx.ws
    cards, version = load_cards(ctx)
    stale = upstream_problems(ctx)
    if stale:
        raise BriefError("cards are out of date with their sources; re-run the affected stages "
                         "before planning:\n  - " + "\n  - ".join(stale))
    existing = read_json(ws.brief_json)
    store = ctx.stages()
    rec = store.get("brief", "plan")
    prev = existing or {}
    header = {
        "title": title if title is not None else prev.get("title") or _default_title(ctx),
        "audience": audience if audience is not None else prev.get("audience", ""),
        "objectives": objectives if objectives is not None else prev.get("objectives", []),
    }
    prereqs = load_prereqs(ctx, cards, version, allow_compute=True)
    fp = _plan_fingerprint(ctx, version, prereqs, header)
    key = hash_obj(fp)
    if existing is not None and not force:
        modified = not rec or hash_obj(existing) != rec.get("brief_hash")
        if rec and rec.get("cache_key") == key:
            log.info("brief: inputs unchanged%s; keeping brief.json",
                     " (manual edits preserved)" if modified else "")
            return render(ctx)
        if modified:
            raise BriefError("brief.json has manual edits but its inputs changed (stale). "
                             "Review it, or run 'brief --force' to re-plan and overwrite it.")

    keep = existing and existing.get("cards_version") == version
    decisions = existing.get("review_decisions", []) if keep else []
    excluded = existing.get("excluded_slide_cards", []) if keep else []
    order = compute_order(cards, version, prereqs, decisions)
    by_id = {c["card_id"]: c for c in cards}
    llm_header = {**header, "language": ctx.settings.lang}
    with model_usage() as used:
        chapters = plan_chapters(ctx, llm_header, [by_id[c] for c in order["order"]])
    assigned = {cid for ch in chapters for cid in ch["card_ids"]}
    brief = {
        **header, "cards_version": version, "chapters": chapters,
        "unassigned_card_ids": [c for c in order["order"] if c not in assigned],
        "review_decisions": decisions, "excluded_slide_cards": excluded,
    }
    write_json(ws.brief_json, brief)
    if ws.approval_file.exists():
        ws.approval_file.unlink()
        log.info("brief re-planned; previous approval invalidated")
    store.put("brief", "plan", {"status": "success", "cache_key": key, "brief_hash": hash_obj(brief),
                                "fingerprint": fp, "models": summarize_models(used),
                                "cards_version": version})
    return render(ctx)


def _default_title(ctx: Context) -> str:
    m = read_json(ctx.ws.sources_manifest) or {}
    titles = [s.get("title") for s in m.get("sources", []) if s.get("title")]
    return titles[0] if titles else "課程"


# ---------------------------------------------------------------- validation

def validate(brief: dict, cards: list[dict], version: str, order: dict) -> dict:
    errors, warnings = [], []
    ids = [c["card_id"] for c in cards]
    idset = set(ids)
    if brief.get("cards_version") != version:
        errors.append(f"cards_version mismatch: brief={brief.get('cards_version')} cards={version}")
    seen_ch, seen_cards, positions = set(), {}, {}
    for ch in brief.get("chapters", []):
        cid = ch.get("chapter_id")
        if not cid or cid in seen_ch:
            errors.append(f"duplicate or empty chapter_id: {cid!r}")
        seen_ch.add(cid)
        if not str(ch.get("title", "")).strip():
            errors.append(f"{cid}: empty title")
        if len(ch.get("card_ids", [])) < 3:
            warnings.append(f"{cid}: fewer than 3 cards (allowed when justified)")
        for card_id in ch.get("card_ids", []):
            if card_id not in idset:
                errors.append(f"{cid}: unknown card_id {card_id}")
            elif card_id in seen_cards:
                errors.append(f"{cid}: duplicate card_id {card_id} (also in {seen_cards[card_id]})")
            else:
                seen_cards[card_id] = cid
                positions[card_id] = len(positions)
    unassigned = [c for c in order["order"] if c not in seen_cards]
    if unassigned:
        errors.append(f"unassigned cards: {', '.join(unassigned)}")
    for e in order["effective_edges"]:
        a, b = e["from"], e["to"]
        if a in positions and b in positions and positions[a] > positions[b]:
            errors.append(f"prerequisite order violated: {a} must come before {b}"
                          f" ({seen_cards[a]} / {seen_cards[b]})")
    for i, d in enumerate(brief.get("review_decisions", [])):
        if not str(d.get("item_id", "")).strip():
            errors.append(f"review_decisions[{i}]: missing item_id")
        if d.get("decision") not in DECISIONS:
            errors.append(f"review_decisions[{i}]: unknown decision {d.get('decision')!r}")
        if not str(d.get("reason", "")).strip():
            errors.append(f"review_decisions[{i}]: reason is required")
        if d.get("decision") == "map_prerequisite" and d.get("target_card_id") not in idset:
            errors.append(f"review_decisions[{i}]: target_card_id must be an existing card")
    unresolved = {u["item_id"]: u for u in order["unresolved_prerequisites"]}
    for i, d in enumerate(brief.get("review_decisions", [])):
        u = unresolved.get(d.get("item_id"))
        if d.get("decision") == "map_prerequisite":
            if u is None:
                errors.append(f"review_decisions[{i}]: map_prerequisite must refer to an unresolved "
                              f"prerequisite item_id")
            elif d.get("target_card_id") == u["card_id"]:
                errors.append(f"review_decisions[{i}]: a prerequisite cannot map to its own card "
                              f"({u['card_id']})")
        if d.get("decision") in ("audience_has", "need_material") and u is None:
            errors.append(f"review_decisions[{i}]: {d.get('decision')} must refer to an unresolved "
                          f"prerequisite item_id")
    seen_ex = set()
    for i, x in enumerate(brief.get("excluded_slide_cards", [])):
        if x.get("card_id") not in idset:
            errors.append(f"excluded_slide_cards[{i}]: unknown card_id {x.get('card_id')!r}")
        if x.get("card_id") in seen_ex:
            errors.append(f"excluded_slide_cards[{i}]: duplicate card_id {x.get('card_id')}")
        seen_ex.add(x.get("card_id"))
        if not str(x.get("reason", "")).strip():
            errors.append(f"excluded_slide_cards[{i}]: reason is required")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "unassigned_card_ids": unassigned}


def approval_blockers(brief: dict, order: dict, validation: dict, report: RunReport,
                      stale: list[str]) -> list[str]:
    blockers = list(validation["errors"]) + [f"stale upstream: {p}" for p in stale]
    decisions = {d.get("item_id"): d for d in brief.get("review_decisions", [])}
    mapped = {(e["item_id"], e["from"], e["to"]) for e in order["mapped_edges"]}
    effective = {(e["from"], e["to"]) for e in order["effective_edges"]}
    for u in order["unresolved_prerequisites"]:
        d = decisions.get(u["item_id"])
        if not d:
            blockers.append(f"unresolved prerequisite needs a review decision: {u['item_id']} "
                            f"({u['card_id']} needs '{u['prerequisite']}')")
        elif d["decision"] == "need_material":
            blockers.append(f"{u['item_id']}: more material required; add it and re-run")
        elif d["decision"] == "map_prerequisite":
            edge = (d.get("target_card_id"), u["card_id"])
            if (u["item_id"], *edge) not in mapped or edge not in effective:
                blockers.append(f"{u['item_id']}: mapping to {edge[0]} does not produce an effective "
                                f"prerequisite edge {edge[0]} -> {edge[1]}")
        elif d["decision"] != "audience_has":
            blockers.append(f"{u['item_id']}: decision must be map_prerequisite, audience_has or need_material")
    for e in order["removed_edges"]:
        blockers.append(f"cycle edge removed automatically and not confirmed: {e['item_id']} "
                        f"(record decision remove_edge, or fix the cycle)")
    for f in report.failures():
        d = decisions.get(f["item_id"])
        if not d or d.get("decision") != "accept_missing":
            blockers.append(f"{f['stage']} unit {f['unit']} {f['status']} ({f['reason']}); "
                            f"accept with item_id {f['item_id']} or fix and re-run")
    for stage in ("ingest", "chunk", "cards", "merge"):
        if stage not in report.data["stages"]:
            blockers.append(f"stage '{stage}' has not been run")
    return blockers


# ---------------------------------------------------------------- rendering

def _current(ctx: Context) -> tuple[dict, list[dict], str, dict, dict]:
    brief = read_json(ctx.ws.brief_json)
    if brief is None:
        raise BriefError("no brief/brief.json; run 'brief' first")
    cards, version = load_cards(ctx)
    prereqs = load_prereqs(ctx, cards, version, allow_compute=False)
    order = compute_order(cards, version, prereqs, brief.get("review_decisions", []))
    return brief, cards, version, prereqs, order


def render(ctx: Context) -> dict:
    """Apply review decisions, rebuild order, validate and regenerate brief.md. No LLM calls."""
    brief, cards, version, _, order = _current(ctx)
    write_json(ctx.ws.order_file, order)
    report = ctx.report()
    validation = validate(brief, cards, version, order)
    stale = upstream_problems(ctx)
    blockers = approval_blockers(brief, order, validation, report, stale)
    write_text(ctx.ws.brief_md, render_markdown(brief, cards, order, validation, blockers, report))
    return {"validation": validation, "blockers": blockers, "stale": stale,
            "partial": bool(report.failures())}


def approve(ctx: Context) -> dict:
    result = render(ctx)
    if result["blockers"]:
        raise BriefError("cannot approve:\n  - " + "\n  - ".join(result["blockers"]))
    brief, _, version, _, order = _current(ctx)
    report = ctx.report()
    failures = report.failures()
    approval = {
        "brief_hash": hash_obj(brief), "cards_version": version, "order_hash": order_hash(order),
        "completeness_hash": report.completeness_hash(),
        "partial": bool(failures), "accepted_failures": [f["item_id"] for f in failures],
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(ctx.ws.approval_file, approval)
    return approval


def verify_approval(ctx: Context) -> dict:
    """Called before any slide request. Raises BriefError if anything drifted."""
    approval = read_json(ctx.ws.approval_file)
    if not approval:
        raise BriefError("no approval: review brief/brief.md and run 'brief --approve'")
    brief, cards, version, _, order = _current(ctx)
    report = ctx.report()
    problems = []
    if hash_obj(brief) != approval.get("brief_hash"):
        problems.append("brief.json changed after approval")
    if version != approval.get("cards_version") or brief.get("cards_version") != version:
        problems.append("cards changed after approval")
    if order_hash(order) != approval.get("order_hash"):
        problems.append("effective order changed after approval")
    stored_order = read_json(ctx.ws.order_file)
    if stored_order is None or order_hash(stored_order) != approval.get("order_hash"):
        problems.append("brief/order.json does not match the approved order")
    if report.completeness_hash() != approval.get("completeness_hash"):
        problems.append("upstream completeness changed after approval")
    validation = validate(brief, cards, version, order)
    blockers = approval_blockers(brief, order, validation, report, upstream_problems(ctx))
    if blockers:
        problems.extend(blockers)
    if problems:
        raise BriefError("approval is not valid; re-review and approve again:\n  - " + "\n  - ".join(problems))
    return {"approval": approval, "brief": brief, "cards": cards, "order": order,
            "failures": report.failures()}


def render_markdown(brief: dict, cards: list[dict], order: dict, validation: dict,
                    blockers: list[str], report: RunReport) -> str:
    by_id = {c["card_id"]: c for c in cards}
    failures = report.failures()
    L = [f"# Course Brief：{brief.get('title', '')}", ""]
    L.append(f"- 受眾：{brief.get('audience') or '（未指定）'}")
    for o in brief.get("objectives", []):
        L.append(f"- 課程學習目標：{o}")
    L.append(f"- cards_version：`{brief.get('cards_version')}`")
    L.append(f"- 狀態：{'驗證通過' if validation['valid'] else '驗證失敗'}"
             f"{'（partial：有失敗或缺漏）' if failures else ''}"
             f"；{'可核准' if not blockers else '尚不可核准'}")
    L.append("")
    L.append("> 本檔為閱讀輸出；請編輯 `brief/brief.json`，再執行 `brief --render-only` 與 `brief --approve`。")
    L.append("")
    if blockers:
        L += ["## 核准前需處理", ""] + [f"- {b}" for b in blockers] + [""]
    if validation["warnings"]:
        L += ["## 提醒", ""] + [f"- {w}" for w in validation["warnings"]] + [""]
    if failures:
        L += ["## 失敗與缺漏", ""]
        L += [f"- `{f['item_id']}` {f['stage']} / {f['unit']}：{f['status']}（{f['reason']}）"
              f"{' ' + f['error'] if f.get('error') else ''}" for f in failures]
        L.append("")
    if order["unresolved_prerequisites"]:
        L += ["## 未解決的先備知識", ""]
        for u in order["unresolved_prerequisites"]:
            best = f"，最接近 {u['best_candidate']}（{u['similarity']}）" if u.get("best_candidate") else ""
            L.append(f"- `{u['item_id']}` {u['card_id']} 需要「{u['prerequisite']}」{best}")
        L.append("")
    if order["cycles"] or order["removed_edges"] or order["human_removed_edges"]:
        L += ["## 先備關係循環", ""]
        for c in order["cycles"]:
            L.append(f"- 循環：{' ↔ '.join(c)}")
        for e in order["removed_edges"]:
            L.append(f"- 草稿暫時移除（未確認）：`{e['item_id']}` 相似度 {e['similarity']}")
        for e in order["human_removed_edges"]:
            L.append(f"- 人工移除：{e['from']} → {e['to']}（{e['reason']}）")
        L.append("")
    excluded = {x.get("card_id"): x.get("reason") for x in brief.get("excluded_slide_cards", [])}
    for ch in brief.get("chapters", []):
        L += [f"## {ch['chapter_id']} {ch['title']}", ""]
        for o in ch.get("learning_objectives", []):
            L.append(f"- 學習目標：{o}")
        for v in ch.get("suggested_visuals", []):
            L.append(f"- 建議圖表：{v}")
        L.append("")
        for cid in ch.get("card_ids", []):
            c = by_id.get(cid)
            if not c:
                L += [f"### {cid}（不存在的卡片）", ""]
                continue
            L += [f"### {cid} {c['concept']}", ""]
            if cid in excluded:
                L.append(f"> 投影片排除：{excluded[cid]}")
                L.append("")
            if c["merge_status"] not in ("single", "merged"):
                L.append(f"> 合併狀態：{c['merge_status']}")
                L.append("")
            L.append(f"**定義**：{c['definition']}")
            L.append("")
            if c["intuition"]:
                L += [f"**直觀理解**：{c['intuition']}", ""]
            for f in c["formulas"]:
                L.append(f"- 公式：$${f['latex']}$$")
            for e in c["examples"]:
                L.append(f"- 例子：{e['text']}")
            for m in c["misconceptions"]:
                L.append(f"- 常見誤解：{m['text']}")
            if c["prerequisites"]:
                L.append(f"- 先備：{'、'.join(c['prerequisites'])}")
            L.append(f"- 合併自：{', '.join(c['merged_from'])}")
            for i, conf in enumerate(c["conflicts"], 1):
                L.append(f"- **來源矛盾 {i}：{conf['topic']}**")
                for cl in conf["claims"]:
                    L.append(f"  - {cl['statement']}（{', '.join(cl['card_ids'])}）")
            L += ["", "<details><summary>證據</summary>", ""]
            for e in c["evidence"]:
                text = e["text"].replace("\n", " ")
                L.append(f"- `{e['source_id']}` {e['locator']}（支持：{', '.join(e['supports']) or '—'}）："
                         f"{text[:300]}{'…' if len(text) > 300 else ''}")
            L += ["", "</details>", ""]
    if validation["unassigned_card_ids"]:
        L += ["## 未分配章節的卡片", ""] + [f"- {c}" for c in validation["unassigned_card_ids"]] + [""]
    if brief.get("review_decisions"):
        L += ["## 審核決定", ""]
        L += [f"- `{d.get('item_id')}` {d.get('decision')}：{d.get('reason')}" for d in brief["review_decisions"]]
        L.append("")
    return "\n".join(L)
