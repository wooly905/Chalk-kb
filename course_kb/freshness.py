"""Upstream dependency-chain verification.

Approval and slide generation must only use cards that still correspond to the current
normalized documents, chunks and source evidence. Each check compares the input version a
stage recorded against the current content of the artifact it consumed."""
from __future__ import annotations

from .hashing import hash_file, hash_json_file, hash_obj
from .io_utils import read_json
from .state import Context, _unit_ok


def upstream_problems(ctx: Context) -> list[str]:
    ws, s = ctx.ws, ctx.settings
    report = ctx.report()
    stages = report.data["stages"]
    problems: list[str] = []
    manifest = read_json(ws.sources_manifest) or {}
    source_ids = [x["id"] for x in manifest.get("sources", [])]
    docs: dict[str, dict] = {}

    ingest = stages.get("ingest", {}).get("units", {})
    if sorted(ingest) != sorted(source_ids):
        problems.append("ingest report does not cover the current source list; run ingest")
    for sid, u in ingest.items():
        if not _unit_ok(u):
            continue
        doc = read_json(ws.normalized / f"{sid}.json")
        if doc is None or doc.get("source_version") != u.get("input_version"):
            problems.append(f"normalized/{sid}.json does not match the ingested source version; run ingest")
            continue
        for g in doc["segments"]:
            if g["kind"] == "frame":
                img = ws.root / g["path"]
                if not img.exists() or hash_file(img) != g["image_hash"]:
                    problems.append(f"frame {g['frame_id']} of {sid} is missing or changed; run ingest")
        docs[sid] = doc

    chunk_units = stages.get("chunk", {}).get("units", {})
    if sorted(chunk_units) != sorted(source_ids):
        problems.append("chunk report does not cover the current source list; run chunk")
    for sid, u in chunk_units.items():
        if not _unit_ok(u) or sid not in docs:
            continue
        if u.get("input_version") != hash_json_file(ws.normalized / f"{sid}.json"):
            problems.append(f"chunks of {sid} were built from an older normalized document; run chunk")

    chunks = (read_json(ws.chunks_file) or {}).get("chunks", [])
    seg_index = {sid: {g["segment_id"]: g for g in d["segments"]} for sid, d in docs.items()}
    for c in chunks:
        doc = docs.get(c["source_id"])
        if doc is None or c["source_version"] != doc["source_version"] or \
                any(g not in seg_index[c["source_id"]] for g in c["segment_ids"]):
            problems.append(f"chunk {c['chunk_id']} does not match the current normalized document; run chunk")

    card_units = stages.get("cards", {}).get("units", {})
    if sorted(card_units) != sorted(c["chunk_id"] for c in chunks):
        problems.append("card extraction does not cover the current chunks; run cards")
    for c in chunks:
        u = card_units.get(c["chunk_id"])
        if u and u.get("input_version") != c["chunk_hash"]:
            problems.append(f"cards of {c['chunk_id']} were extracted from an older chunk; run cards")

    raw = (read_json(ws.raw_cards_file) or {}).get("cards", [])
    cards_doc = read_json(ws.cards_file) or {}
    built = cards_doc.get("built_from", {})
    if built.get("raw_cards_hash") != hash_obj(raw):
        problems.append("cards/cards.json was not built from the current raw cards; run cards")
    if built.get("merge_similarity") != s.merge_similarity:
        problems.append("merge_similarity changed since cards were merged; run cards")

    stale_evidence = set()
    for card in raw + cards_doc.get("cards", []):
        for e in card.get("evidence", []):
            seg = seg_index.get(e["source_id"], {}).get(e["segment_id"])
            doc = docs.get(e["source_id"])
            if seg is None or doc is None or doc["source_version"] != e["source_version"] \
                    or seg["text"] != e["text"]:
                stale_evidence.add(card["card_id"])
    if stale_evidence:
        problems.append(f"evidence no longer matches the current sources for cards: "
                        f"{', '.join(sorted(stale_evidence))}; re-run chunk and cards")
    return list(dict.fromkeys(problems))
