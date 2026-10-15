"""chunk: split normalized documents into LLM-sized blocks. No LLM calls."""
from __future__ import annotations

import logging
import re

from .hashing import hash_json_file, hash_obj
from .ingest import load_manifest
from .io_utils import read_json, write_json
from .llm import describe_error
from .normalize import fmt_time
from .state import Context, _unit_ok

log = logging.getLogger("course_kb")
PROCESSOR_VERSION = "chunk-1"
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
_WORD = re.compile(r"[A-Za-z0-9_]+(?:['’-][A-Za-z0-9_]+)*")


def estimate_tokens(text: str) -> int:
    """CJK characters ~1 token each; other words ~1.3 tokens each."""
    cjk = len(_CJK.findall(text))
    words = len(_WORD.findall(_CJK.sub(" ", text)))
    return int(round(cjk + words * 1.3))


def render_segment(seg: dict) -> str:
    kind = seg["kind"]
    if kind == "heading":
        return "#" * seg.get("level", 1) + " " + seg["text"]
    if kind == "transcript":
        return f"[{fmt_time(seg['start'])}] {seg['text']}"
    if kind == "frame":
        return f"［畫面 {fmt_time(seg['start'])}］ {seg['text']}"
    return seg["text"]


def chunk_document(doc: dict, max_tokens: int, window_sec: int) -> list[dict]:
    is_transcript = doc["type"] in ("subtitle", "media", "youtube")
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_tokens = 0
    for seg in doc["segments"]:
        t = estimate_tokens(render_segment(seg))
        cut = False
        if cur and cur_tokens + t > max_tokens:
            cut = True
        elif cur and is_transcript and "start" in seg and "start" in cur[0]:
            if seg.get("end", seg["start"]) - cur[0]["start"] > window_sec:
                cut = True
        elif (cur and not is_transcript and seg["kind"] == "heading"
              and seg.get("level", 9) <= 2 and cur_tokens > 0.3 * max_tokens):
            cut = True
        if cut:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(seg)
        cur_tokens += t
    if cur:
        chunks.append(cur)

    out = []
    for segs in chunks:
        if all(s["kind"] == "heading" for s in segs):
            continue
        text = "\n\n".join(render_segment(s) for s in segs)
        out.append({
            "chunk_id": f"{doc['source_id']}-{len(out) + 1:03d}",
            "source_id": doc["source_id"],
            "source_version": doc["source_version"],
            "segment_ids": [s["segment_id"] for s in segs],
            "text": text,
            "token_estimate": estimate_tokens(text),
        })
    for c in out:
        c["chunk_hash"] = hash_obj({k: c[k] for k in ("chunk_id", "source_version", "segment_ids", "text")})
    return out


def run_chunk(ctx: Context) -> str:
    ws, s = ctx.ws, ctx.settings
    sources = load_manifest(ctx)
    ids = [x["id"] for x in sources]
    store, report = ctx.stages(), ctx.report()
    ingest_units = report.data["stages"].get("ingest", {}).get("units", {})
    report.start_stage("chunk", ids)
    store.drop_missing("chunk", ids)
    all_chunks = []
    for sid in ids:
        out = ws.p("chunks", "by_source", f"{sid}.json")
        norm = ws.normalized / f"{sid}.json"
        iu = ingest_units.get(sid)
        if not iu or not _unit_ok(iu) or not norm.exists():
            report.set_unit("chunk", sid, "skipped", "missing_input", None)
            continue
        try:
            doc_hash = hash_json_file(norm)
            fp = {"doc": doc_hash, "max": s.chunk_max_tokens,
                  "window": s.transcript_window_sec, "processor": PROCESSOR_VERSION}
            key = hash_obj(fp)
            if store.is_valid("chunk", sid, key, [out]):
                chunks = read_json(out)["chunks"]
                report.set_unit("chunk", sid, "skipped", "cache_hit", doc_hash)
            else:
                doc = read_json(norm)
                chunks = chunk_document(doc, s.chunk_max_tokens, s.transcript_window_sec)
                write_json(out, {"source_id": sid, "chunks": chunks})
                store.record_success("chunk", sid, key, doc_hash, [out], fingerprint=fp)
                report.set_unit("chunk", sid, "success", "", doc_hash)
            all_chunks.extend(chunks)
        except Exception as e:  # noqa: BLE001
            report.set_unit("chunk", sid, "failed", type(e).__name__, None, describe_error(e))
            log.error("chunk %s failed: %s", sid, e)
    status = report.finish_stage("chunk")
    write_json(ws.chunks_file, {"completeness": report.artifact_completeness(("ingest", "chunk")),
                                "chunks": all_chunks})
    log.info("chunk: %d chunks (%s)", len(all_chunks), status)
    return status
