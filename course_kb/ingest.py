"""ingest: fetch each source into raw/ and convert it into normalized/{id}.json."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .hashing import hash_obj, sha256_bytes
from .io_utils import read_json, write_bytes, write_json
from .llm import FatalLLMError, describe_error
from .normalize import (cues_to_segments, guess_language, markdown_title, parse_markdown,
                        parse_subtitles)
from .sources import load_sources
from .state import Context, model_usage

log = logging.getLogger("course_kb")
PROCESSOR_VERSION = "ingest-2"
LOCAL_TYPES = ("text", "subtitle", "media")
AV_TYPES = ("media", "youtube")


def _config_fingerprint(src: dict) -> dict:
    return {k: src.get(k) for k in ("type", "location", "title", "language", "clean", "keyframes",
                                     "auto_captions", "rights_confirmed")}


def _local_input_hash(ctx: Context, src: dict) -> str:
    """Cheap pre-processing version of a local source, including settings that shape its output."""
    if src["type"] == "media":
        from .media import media_input_hash
        return media_input_hash(ctx, src, _config_fingerprint(src))
    extra = None
    if src["type"] == "subtitle" and src.get("auto_captions") and src.get("clean"):
        from .media import clean_fingerprint
        extra = clean_fingerprint(ctx)
    return hash_obj({"raw": sha256_bytes(_read_local(src)), "config": _config_fingerprint(src),
                     "clean": extra})


def _cache_ok(ctx: Context, store, sid: str, key: str, out: Path) -> bool:
    """Outer cache hit: record matches and every required output (incl. frame images) is intact."""
    if not store.is_valid("ingest", sid, key, [out]):
        return False
    from .media import frame_outputs_valid
    return frame_outputs_valid(ctx, read_json(out) or {})


def _read_local(src: dict) -> bytes:
    p = Path(src["location"])
    if not p.exists():
        raise FileNotFoundError(f"source file not found: {p}")
    return p.read_bytes()


def _fetch(src: dict, ctx: Context) -> tuple[bytes, str]:
    """Return (raw bytes, raw file suffix)."""
    t = src["type"]
    if t == "text":
        return _read_local(src), Path(src["location"]).suffix or ".txt"
    if t == "subtitle":
        return _read_local(src), Path(src["location"]).suffix or ".srt"
    if t == "web":
        try:
            import trafilatura
        except ImportError as e:
            raise RuntimeError("web sources require 'trafilatura' (pip install trafilatura)") from e
        html = trafilatura.fetch_url(src["location"])
        if not html:
            raise RuntimeError("無法從網址抽出正文：download failed")
        return html.encode("utf-8") if isinstance(html, str) else html, ".html"
    if t in ("youtube", "media"):  # pragma: no cover - handled by media.ingest_av
        raise ValueError(f"{t} sources are ingested by course_kb.media")
    raise ValueError(f"unknown type {t}")


def _normalize(ctx: Context, src: dict, raw: bytes, source_version: str, force: bool = False) -> dict:
    sid, t = src["id"], src["type"]
    if src.get("keyframes"):
        raise ValueError("keyframes only apply to media and youtube sources")
    if t == "text":
        text = raw.decode("utf-8-sig")
        segments = parse_markdown(sid, text)
        title = src.get("title") or markdown_title(text) or Path(src["location"]).stem
        origin = "text"
    elif t == "subtitle":
        text = raw.decode("utf-8-sig")
        origin = "auto_subtitle" if src.get("auto_captions") else "manual_subtitle"
        segments = cues_to_segments(sid, parse_subtitles(text), origin=origin)
        if src.get("clean") and origin == "auto_subtitle" and segments:
            from .media import proofread
            segments, _ = proofread(ctx, sid, segments, ctx.ws.raw / f"{sid}.clean.json", force)
        title = src.get("title") or Path(src["location"]).stem
    elif t == "web":
        import trafilatura

        html = raw.decode("utf-8", errors="replace")
        text = trafilatura.extract(html, output_format="markdown", include_formatting=True,
                                   include_comments=False) or ""
        if not text.strip():
            raise RuntimeError("無法從網址抽出正文：no main content extracted")
        meta = trafilatura.extract_metadata(html)
        segments = parse_markdown(sid, text)
        title = src.get("title") or (meta.title if meta and meta.title else None) or src["location"]
        origin = "web"
    else:  # pragma: no cover - guarded by _fetch
        raise ValueError(t)
    if not segments:
        raise RuntimeError("source produced no segments")
    full = "\n".join(s["text"] for s in segments)
    return {
        "source_id": sid, "source_version": source_version, "type": t, "title": title,
        "location": src["location"], "license": src.get("license"),
        "language": src.get("language") or guess_language(full),
        "text_origin": origin, "segments": segments,
    }


def run_ingest(ctx: Context, sources_path: Path, only: Optional[list[str]] = None,
               force: bool = False) -> str:
    sources = load_sources(sources_path)
    ids = [s["id"] for s in sources]
    unknown = set(only or []) - set(ids)
    if unknown:
        raise ValueError(f"--only refers to unknown source ids: {', '.join(sorted(unknown))}")
    ws = ctx.ws
    write_json(ws.sources_manifest, {"sources": [
        {k: s.get(k) for k in ("id", "type", "location", "title", "license", "language")}
        for s in sources]})
    store, report = ctx.stages(), ctx.report()
    previous = report.data["stages"].get("ingest", {}).get("units", {})
    report.start_stage("ingest", ids)
    store.drop_missing("ingest", ids)

    for src in sources:
        sid = src["id"]
        out = ws.normalized / f"{sid}.json"
        selected = not only or sid in only
        rec = store.get("ingest", sid)
        if not selected:
            # Not processed this run: only trust it if its content can be re-verified cheaply.
            if src["type"] in LOCAL_TYPES and rec and out.exists():
                try:
                    input_hash = _local_input_hash(ctx, src)
                except FileNotFoundError:
                    input_hash = None
                key = hash_obj({"input": input_hash, "processor": PROCESSOR_VERSION})
                if input_hash and _cache_ok(ctx, store, sid, key, out):
                    report.set_unit("ingest", sid, "skipped", "cache_hit", input_hash)
                    continue
            prev = previous.get(sid, {})
            report.set_unit("ingest", sid, "skipped", "not_checked", prev.get("input_version"))
            continue
        input_hash = None
        fingerprint = {"config": _config_fingerprint(src), "processor": PROCESSOR_VERSION}
        try:
            if src["type"] == "youtube":
                # The version depends on the subtitle/audio/frames actually retrieved.
                from .media import ingest_av
                with model_usage() as used:
                    doc = ingest_av(ctx, ctx.tools, src, _config_fingerprint(src), force=force)
                input_hash = doc["source_version"]
                key = hash_obj({"input": input_hash, "processor": PROCESSOR_VERSION})
                if not force and _cache_ok(ctx, store, sid, key, out) and read_json(out) == doc:
                    report.set_unit("ingest", sid, "skipped", "cache_hit", input_hash)
                    continue
            else:
                if src["type"] in LOCAL_TYPES:
                    input_hash = _local_input_hash(ctx, src)
                    key = hash_obj({"input": input_hash, "processor": PROCESSOR_VERSION})
                    if not force and _cache_ok(ctx, store, sid, key, out):
                        report.set_unit("ingest", sid, "skipped", "cache_hit", input_hash)
                        log.info("ingest %s: cache hit", sid)
                        continue
                with model_usage() as used:
                    if src["type"] == "media":
                        from .media import ingest_av
                        doc = ingest_av(ctx, ctx.tools, src, _config_fingerprint(src), input_hash, force=force)
                    else:
                        raw, suffix = _fetch(src, ctx)
                        if src["type"] == "web":
                            input_hash = hash_obj({"raw": sha256_bytes(raw), "config": _config_fingerprint(src)})
                            key = hash_obj({"input": input_hash, "processor": PROCESSOR_VERSION})
                            if not force and _cache_ok(ctx, store, sid, key, out):
                                report.set_unit("ingest", sid, "skipped", "cache_hit", input_hash)
                                continue
                        write_bytes(ws.raw / f"{sid}{suffix}", raw)
                        doc = _normalize(ctx, src, raw, input_hash, force)
            write_json(out, doc)
            store.record_success("ingest", sid, key, input_hash, [out], fingerprint=fingerprint, models=used)
            report.set_unit("ingest", sid, "success", doc.get("text_origin", ""), input_hash)
            log.info("ingest %s: %d segments", sid, len(doc["segments"]))
        except Exception as e:  # noqa: BLE001 - every failure is recorded per unit
            report.set_unit("ingest", sid, "failed", getattr(e, "kind", type(e).__name__),
                            input_hash, describe_error(e))
            store.record_failure("ingest", sid, "", input_hash or "", describe_error(e))
            log.error("ingest %s failed: %s", sid, e)
            if isinstance(e, FatalLLMError):
                raise
    return report.finish_stage("ingest")


def load_manifest(ctx: Context) -> list[dict]:
    data = read_json(ctx.ws.sources_manifest)
    if not data:
        raise RuntimeError("no source manifest; run 'ingest' first")
    return data["sources"]
