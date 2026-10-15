"""cards: extract knowledge cards per chunk, then cluster and merge them across sources."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .chunk import estimate_tokens
from .embed import average_linkage_clusters
from .hashing import hash_obj
from .io_utils import read_json, write_json
from .llm import (SPLITTABLE, FatalLLMError, LLMError, SchemaValidationError, describe_error)
from .prompts import (CARD_FIELDS, EXTRACT_SCHEMA, MERGE_SCHEMA, PROMPT_VERSIONS,
                      extract_instructions, merge_instructions)
from .state import Context, StageStore, RunReport, model_usage

log = logging.getLogger("course_kb")
EXTRACT_PROCESSOR = "extract-1"
MERGE_PROCESSOR = "merge-1"
MAX_EXAMPLES = 5


class EvidenceError(SchemaValidationError):
    kind = "invalid_evidence"


def _llm_fingerprint(ctx: Context, role: str, prompt: str, schema: dict) -> dict:
    s = ctx.settings
    return {"prompt": PROMPT_VERSIONS[prompt], "schema": hash_obj(schema), "base_url": s.base_url,
            "model": s.model_for(role), "lang": s.lang}


# ---------------------------------------------------------------- extraction

def _load_segments(ctx: Context, source_ids: set[str]) -> tuple[dict, dict]:
    segs, docs = {}, {}
    for sid in source_ids:
        doc = read_json(ctx.ws.normalized / f"{sid}.json")
        if doc is None:
            continue
        docs[sid] = doc
        for seg in doc["segments"]:
            segs[seg["segment_id"]] = {**seg, "source_id": sid, "source_version": doc["source_version"]}
    return segs, docs


def evidence_record(seg: dict, supports: list[str]) -> dict:
    rec = {"segment_id": seg["segment_id"], "source_id": seg["source_id"],
           "source_version": seg["source_version"], "kind": seg["kind"],
           "locator": seg["locator"], "text": seg["text"], "supports": sorted(set(supports))}
    if seg.get("raw_text") and seg["raw_text"] != seg["text"]:
        rec["raw_text"] = seg["raw_text"]
    if seg["kind"] == "frame":
        rec["frame"] = {k: seg.get(k) for k in ("frame_id", "path", "start", "image_hash")}
    return rec


def _extract_segments(ctx: Context, chunk: dict, seg_ids: list[str], segs: dict) -> list[dict]:
    """Extract cards from a list of segments; on truncation split the list in half."""
    alias = {f"S{i + 1}": sid for i, sid in enumerate(seg_ids)}
    payload = {"chunk_id": chunk["chunk_id"], "source_id": chunk["source_id"],
               "segments": [{"segment_ref": a, "kind": segs[sid]["kind"],
                             "locator": segs[sid]["locator"], "text": segs[sid]["text"]}
                            for a, sid in alias.items()]}
    try:
        data, _ = ctx.llm.structured("text", "extract_cards", EXTRACT_SCHEMA,
                                     extract_instructions(ctx.settings.lang),
                                     json.dumps(payload, ensure_ascii=False),
                                     ctx.settings.output_tokens)
    except SPLITTABLE:
        if len(seg_ids) <= 1:
            raise
        mid = len(seg_ids) // 2
        log.warning("extract %s too large; splitting %d segments", chunk["chunk_id"], len(seg_ids))
        return (_extract_segments(ctx, chunk, seg_ids[:mid], segs)
                + _extract_segments(ctx, chunk, seg_ids[mid:], segs))
    cards = []
    for c in data["cards"]:
        if not c["evidence"]:
            raise EvidenceError(f"card '{c['concept']}' has no evidence")
        supports: dict[str, set] = {}
        for ev in c["evidence"]:
            sid = alias.get(ev["segment_ref"])
            if sid is None:
                raise EvidenceError(f"card '{c['concept']}' cites unknown segment_ref {ev['segment_ref']}")
            supports.setdefault(sid, set()).update(ev["fields"])
        cards.append({**{k: c[k] for k in CARD_FIELDS}, "_supports": supports})
    return cards


def _finalize_raw_cards(chunk: dict, extracted: list[dict], segs: dict) -> list[dict]:
    out = []
    for i, c in enumerate(extracted, 1):
        ordered = [sid for sid in chunk["segment_ids"] if sid in c["_supports"]]
        evidence = [evidence_record(segs[sid], sorted(c["_supports"][sid])) for sid in ordered]
        card = {"card_id": f"{chunk['chunk_id']}-c{i:02d}", "chunk_id": chunk["chunk_id"],
                **{k: c[k] for k in CARD_FIELDS},
                "evidence_segment_ids": ordered, "evidence": evidence,
                "sources": _sources_of(evidence), "visual_refs": _visuals_of(evidence)}
        out.append(card)
    return out


def _sources_of(evidence: list[dict]) -> list[dict]:
    seen, out = set(), []
    for e in evidence:
        k = (e["source_id"], e["locator"])
        if k not in seen:
            seen.add(k)
            out.append({"source_id": e["source_id"], "locator": e["locator"]})
    return out


def _visuals_of(evidence: list[dict]) -> list[dict]:
    return [{**e["frame"], "source_id": e["source_id"], "description": e["text"]}
            for e in evidence if e.get("frame")]


def _extract_unit(ctx: Context, chunk: dict, segs: dict, store: StageStore, report: RunReport,
                  force: bool) -> Optional[list[dict]]:
    cid = chunk["chunk_id"]
    out = ctx.ws.extract_dir / f"{cid}.json"
    fp = {"chunk": chunk["chunk_hash"], "processor": EXTRACT_PROCESSOR,
          **_llm_fingerprint(ctx, "text", "extract", EXTRACT_SCHEMA)}
    key = hash_obj(fp)
    if not force and store.is_valid("cards", cid, key, [out]):
        report.set_unit("cards", cid, "skipped", "cache_hit", chunk["chunk_hash"])
        return read_json(out)["cards"]
    try:
        missing = [s for s in chunk["segment_ids"] if s not in segs]
        if missing:
            raise RuntimeError(f"chunk references {len(missing)} unknown segments")
        with model_usage() as used:
            cards = _finalize_raw_cards(chunk, _extract_segments(ctx, chunk, chunk["segment_ids"], segs), segs)
        write_json(out, {"chunk_id": cid, "chunk_hash": chunk["chunk_hash"], "cards": cards})
        store.record_success("cards", cid, key, chunk["chunk_hash"], [out], {"card_count": len(cards)},
                             fingerprint=fp, models=used)
        report.set_unit("cards", cid, "success", f"{len(cards)} cards", chunk["chunk_hash"])
        return cards
    except Exception as e:  # noqa: BLE001 - recorded as unit failure; other chunks continue
        store.record_failure("cards", cid, key, chunk["chunk_hash"], describe_error(e))
        report.set_unit("cards", cid, "failed", getattr(e, "kind", type(e).__name__),
                        chunk["chunk_hash"], describe_error(e))
        log.error("extract %s failed: %s", cid, e)
        if isinstance(e, FatalLLMError):
            raise
        return None


# ---------------------------------------------------------------- merge

def _item_from_card(card: dict) -> dict:
    cid = card["card_id"]
    return {"concept": card["concept"], "concept_from": [cid],
            "definition": card["definition"], "definition_from": [cid],
            "intuition": card["intuition"], "intuition_from": [cid],
            "examples": [{"text": t, "card_ids": [cid]} for t in card["examples"]],
            "formulas": [{"latex": t, "card_ids": [cid]} for t in card["formulas"]],
            "misconceptions": [{"text": t, "card_ids": [cid]} for t in card["misconceptions"]],
            "prerequisites": list(card["prerequisites"]), "conflicts": [], "members": [cid]}


def _item_payload(ref: str, item: dict) -> dict:
    """Includes previously recorded conflicting claims so cross-batch integration can compare
    every known statement, not only the chosen definition."""
    return {"ref": ref, "concept": item["concept"], "definition": item["definition"],
            "intuition": item["intuition"], "examples": [e["text"] for e in item["examples"]],
            "formulas": [f["latex"] for f in item["formulas"]],
            "misconceptions": [m["text"] for m in item["misconceptions"]],
            "prerequisites": item["prerequisites"],
            "known_conflicts": [{"topic": c["topic"], "claims": [cl["statement"] for cl in c["claims"]]}
                                for c in item["conflicts"]]}


class NotSameConcept(Exception):
    pass


def _union(*lists) -> list:
    out = []
    for lst in lists:
        for x in lst:
            if x not in out:
                out.append(x)
    return out


def _merge_call(ctx: Context, items: list[dict]) -> dict:
    alias = {f"C{i + 1}": it for i, it in enumerate(items)}
    payload = {"cards": [_item_payload(a, it) for a, it in alias.items()]}
    data, _ = ctx.llm.structured("text", "merge_cards", MERGE_SCHEMA,
                                 merge_instructions(ctx.settings.lang),
                                 json.dumps(payload, ensure_ascii=False), ctx.settings.output_tokens)
    if not data["is_same_concept"]:
        raise NotSameConcept()

    def refs(lst: list[str]) -> list[dict]:
        bad = [r for r in lst if r not in alias]
        if bad:
            raise SchemaValidationError(f"merge cites unknown cards {bad}")
        return [alias[r] for r in lst]

    def attr_list(field: str, key: str, entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            src_items = refs(e["from"])
            if not src_items:
                raise SchemaValidationError(f"{field} entry without 'from'")
            ids: list[str] = []
            for it in src_items:
                match = [x for x in it[field] if x[key] == e[key]]
                ids = _union(ids, *(m["card_ids"] for m in match)) if match else _union(ids, it["members"])
            out.append({key: e[key], "card_ids": ids})
        return out

    def field_from(field: str, lst: list[str]) -> list[str]:
        items_ = refs(lst)
        if not items_:
            raise SchemaValidationError(f"{field}_from is empty")
        return _union(*(it[f"{field}_from"] for it in items_))

    conflicts = _union(*(it["conflicts"] for it in items))
    for c in data["conflicts"]:
        claims = [{"statement": cl["statement"],
                   "card_ids": _union(*(it["members"] for it in refs(cl["from"])))}
                  for cl in c["claims"]]
        conflicts.append({"topic": c["topic"], "claims": claims})
    return {
        "concept": data["concept"], "concept_from": _union(*(it["members"] for it in items)),
        "definition": data["definition"], "definition_from": field_from("definition", data["definition_from"]),
        "intuition": data["intuition"], "intuition_from": field_from("intuition", data["intuition_from"]),
        "examples": attr_list("examples", "text", data["examples"])[:MAX_EXAMPLES],
        "formulas": attr_list("formulas", "latex", data["formulas"]),
        "misconceptions": attr_list("misconceptions", "text", data["misconceptions"]),
        "prerequisites": _union(data["prerequisites"]),
        "conflicts": conflicts,
        "members": _union(*(it["members"] for it in items)),
    }


def _batches(items: list[dict], budget: int) -> list[list[dict]]:
    out, cur, cur_t = [], [], 0
    for it in items:
        t = estimate_tokens(json.dumps(_item_payload("C0", it), ensure_ascii=False))
        if cur and cur_t + t > budget:
            out.append(cur)
            cur, cur_t = [], 0
        cur.append(it)
        cur_t += t
    if cur:
        out.append(cur)
    return out


def merge_items(ctx: Context, items: list[dict], budget: int) -> dict:
    """Merge a group in budget-sized batches, then integrate batch results. Raises on failure;
    never returns a partial merge that looks complete."""
    while len(items) > 1:
        batches = _batches(items, budget)
        if len(batches) == len(items) and len(items) > 1 and len(batches) > 1:
            # Every item alone exceeds the budget: pair them anyway, the model limit decides.
            batches = [items[i:i + 2] for i in range(0, len(items), 2)]
        results = []
        for b in batches:
            if len(b) == 1:
                results.append(b[0])
                continue
            try:
                results.append(_merge_call(ctx, b))
            except SPLITTABLE:
                if len(b) <= 2:
                    raise
                half = len(b) // 2
                log.warning("merge batch of %d too large; splitting", len(b))
                results.append(merge_items(ctx, b[:half], budget))
                results.append(merge_items(ctx, b[half:], budget))
        if len(results) >= len(items):
            raise LLMError("merge could not reduce the group within budget")
        items = results
    return items[0]


def _card_from_item(item: dict, raw_by_id: dict, status: str) -> dict:
    members = item["members"]
    ev_by_id: dict[str, dict] = {}
    for rid in members:
        for e in raw_by_id[rid]["evidence"]:
            if e["segment_id"] in ev_by_id:
                prev = ev_by_id[e["segment_id"]]
                prev["supports"] = sorted(set(prev["supports"]) | set(e["supports"]))
                prev["card_ids"] = _union(prev["card_ids"], [rid])
            else:
                ev_by_id[e["segment_id"]] = {**e, "card_ids": [rid]}
    evidence = list(ev_by_id.values())

    def ev_ids(card_ids: list[str]) -> list[str]:
        return _union(*(raw_by_id[c]["evidence_segment_ids"] for c in card_ids))

    conflicts = [{"topic": c["topic"],
                  "claims": [{**cl, "evidence_segment_ids": ev_ids(cl["card_ids"])} for cl in c["claims"]]}
                 for c in item["conflicts"]]
    return {
        "card_id": "", "concept": item["concept"], "definition": item["definition"],
        "intuition": item["intuition"], "examples": item["examples"], "formulas": item["formulas"],
        "misconceptions": item["misconceptions"], "prerequisites": item["prerequisites"],
        "field_support": {"concept": item["concept_from"], "definition": item["definition_from"],
                          "intuition": item["intuition_from"]},
        "evidence_segment_ids": [e["segment_id"] for e in evidence], "evidence": evidence,
        "sources": _sources_of(evidence),
        "visual_refs": _union(*(raw_by_id[r]["visual_refs"] for r in members)),
        "conflicts": conflicts, "merged_from": members, "merge_status": status,
    }


def run_merge(ctx: Context, raw_cards: list[dict], force: bool = False) -> str:
    s = ctx.settings
    store, report = ctx.stages(), ctx.report()
    raw_by_id = {c["card_id"]: c for c in raw_cards}
    index = {c["card_id"]: i for i, c in enumerate(raw_cards)}
    groups: list[list[dict]] = []
    if raw_cards:
        texts = [f"{c['concept']}：{c['definition'][:300]}" for c in raw_cards]
        vecs = ctx.embedder.encode(texts)
        groups = [[raw_cards[i] for i in g] for g in average_linkage_clusters(vecs, s.merge_similarity)]
    multi = [g for g in groups if len(g) > 1]
    unit_of = {id(g): "g-" + hash_obj([c["card_id"] for c in g])[:12] for g in multi}
    report.start_stage("merge", unit_of.values())
    store.drop_missing("merge", unit_of.values())

    def do(group: list[dict]) -> list[tuple[dict, str]]:
        if len(group) == 1:
            return [(_item_from_card(group[0]), "single")]
        unit = unit_of[id(group)]
        content = hash_obj([{k: c[k] for k in c if k != "chunk_id"} for c in group])
        fp = {"cards": content, "budget": s.merge_batch_tokens, "processor": MERGE_PROCESSOR,
              **_llm_fingerprint(ctx, "text", "merge", MERGE_SCHEMA)}
        key = hash_obj(fp)
        out = ctx.ws.merge_dir / f"{unit}.json"
        if not force and store.is_valid("merge", unit, key, [out]):
            report.set_unit("merge", unit, "skipped", "cache_hit", content)
            cached = read_json(out)
            return [(it, st) for it, st in zip(cached["items"], cached["statuses"])]
        try:
            with model_usage() as used:
                try:
                    result = [(merge_items(ctx, [_item_from_card(c) for c in group], s.merge_batch_tokens),
                               "merged")]
                except NotSameConcept:
                    result = [(_item_from_card(c), "not_same_concept") for c in group]
            write_json(out, {"unit": unit, "items": [r[0] for r in result],
                             "statuses": [r[1] for r in result]})
            store.record_success("merge", unit, key, content, [out], fingerprint=fp, models=used)
            report.set_unit("merge", unit, "success", result[0][1], content)
            return result
        except Exception as e:  # noqa: BLE001
            store.record_failure("merge", unit, key, content, describe_error(e))
            report.set_unit("merge", unit, "failed", getattr(e, "kind", type(e).__name__),
                            content, describe_error(e))
            log.error("merge %s failed: %s", unit, e)
            if isinstance(e, FatalLLMError):
                raise
            return [(_item_from_card(c), "merge_failed") for c in group]

    with ThreadPoolExecutor(max_workers=max(1, s.max_workers)) as pool:
        results = list(pool.map(do, groups))

    merged = [(item, st) for r in results for item, st in r]
    merged.sort(key=lambda x: min(index[m] for m in x[0]["members"]))
    cards = []
    for n, (item, st) in enumerate(merged, 1):
        card = _card_from_item(item, raw_by_id, st)
        card["card_id"] = f"k{n:03d}"
        cards.append(card)
    status = report.finish_stage("merge")
    write_json(ctx.ws.cards_file, {
        "cards_version": cards_version(cards),
        # Binds the merged cards to the exact raw cards and settings they were built from.
        "built_from": {"raw_cards_hash": hash_obj(raw_cards), "merge_similarity": s.merge_similarity},
        "completeness": report.artifact_completeness(),
        "cards": cards,
    })
    log.info("merge: %d raw cards -> %d cards (%s)", len(raw_cards), len(cards), status)
    return status


def cards_version(cards: list[dict]) -> str:
    return hash_obj(cards)


def run_cards(ctx: Context, force: bool = False) -> str:
    chunks = (read_json(ctx.ws.chunks_file) or {}).get("chunks")
    if chunks is None:
        raise RuntimeError("no chunks; run 'chunk' first")
    store, report = ctx.stages(), ctx.report()
    ids = [c["chunk_id"] for c in chunks]
    report.start_stage("cards", ids)
    store.drop_missing("cards", ids)
    segs, _ = _load_segments(ctx, {c["source_id"] for c in chunks})
    with ThreadPoolExecutor(max_workers=max(1, ctx.settings.max_workers)) as pool:
        results = list(pool.map(lambda c: _extract_unit(ctx, c, segs, store, report, force), chunks))
    raw_cards = [card for r in results if r for card in r]
    extract_status = report.finish_stage("cards")
    write_json(ctx.ws.raw_cards_file, {"completeness": report.artifact_completeness(
        ("ingest", "chunk", "cards")), "cards": raw_cards})
    log.info("cards: %d raw cards from %d chunks (%s)", len(raw_cards), len(chunks), extract_status)
    merge_status = run_merge(ctx, raw_cards, force)
    # Upstream gaps (ingest/chunk) make every downstream artifact partial too.
    upstream_ok = all(report.stage_status(s) == "complete" for s in ("ingest", "chunk"))
    return "complete" if upstream_ok and extract_status == merge_status == "complete" else "partial"
