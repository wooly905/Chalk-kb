"""Workspace layout, per-unit cache records (meta/stages.json) and run report (reports/run.json)."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import Settings
from .hashing import hash_file, hash_obj
from .io_utils import read_json, write_json

log = logging.getLogger("course_kb")

# Stages whose outcome feeds the completeness summary bound into the approval.
UPSTREAM_STAGES = ("ingest", "chunk", "cards", "merge")


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)

    def p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    raw = property(lambda self: self.p("raw"))
    normalized = property(lambda self: self.p("normalized"))
    chunks_file = property(lambda self: self.p("chunks", "chunks.json"))
    raw_cards_file = property(lambda self: self.p("cards", "raw_cards.json"))
    cards_file = property(lambda self: self.p("cards", "cards.json"))
    extract_dir = property(lambda self: self.p("cards", "extract"))
    merge_dir = property(lambda self: self.p("cards", "merge"))
    order_file = property(lambda self: self.p("brief", "order.json"))
    prereqs_file = property(lambda self: self.p("brief", "prereqs.json"))
    brief_json = property(lambda self: self.p("brief", "brief.json"))
    brief_md = property(lambda self: self.p("brief", "brief.md"))
    approval_file = property(lambda self: self.p("brief", "approval.json"))
    stages_file = property(lambda self: self.p("meta", "stages.json"))
    sources_manifest = property(lambda self: self.p("meta", "sources.json"))
    run_file = property(lambda self: self.p("reports", "run.json"))
    outline_file = property(lambda self: self.p("slides", "outline.json"))
    slide_items = property(lambda self: self.p("slides", "items"))
    slides_json = property(lambda self: self.p("slides", "slides.json"))
    slides_md = property(lambda self: self.p("slides", "slides.md"))


class StageStore:
    """Per-unit cache records. A cache hit requires identical cache_key, a previous success,
    and output files whose content hash still matches what was recorded."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self._lock = threading.Lock()
        self.data: dict = read_json(ws.stages_file, {}) or {}

    def get(self, stage: str, unit: str) -> Optional[dict]:
        return self.data.get(stage, {}).get(unit)

    def put(self, stage: str, unit: str, record: dict) -> None:
        with self._lock:
            self.data.setdefault(stage, {})[unit] = record
            write_json(self.ws.stages_file, self.data)

    def drop_missing(self, stage: str, keep: Iterable[str]) -> None:
        keep = set(keep)
        with self._lock:
            units = self.data.get(stage, {})
            for k in [k for k in units if k not in keep]:
                del units[k]
            write_json(self.ws.stages_file, self.data)

    def is_valid(self, stage: str, unit: str, cache_key: str, outputs: Iterable[Path]) -> bool:
        rec = self.get(stage, unit)
        if not rec or rec.get("status") != "success" or rec.get("cache_key") != cache_key:
            return False
        recorded = rec.get("output_hashes", {})
        for path in outputs:
            path = Path(path)
            if not path.exists():
                return False
            if recorded.get(path.name) != hash_file(path):
                return False
        return True

    def record_success(self, stage: str, unit: str, cache_key: str, input_hash: str,
                       outputs: Iterable[Path], extra: Optional[dict] = None,
                       fingerprint: Optional[dict] = None, models: Optional[list] = None) -> None:
        """fingerprint: the auditable inputs of cache_key (settings, prompt/schema/processor
        versions, API base URL, requested model) — never secrets. models: requested and
        actual models returned by the service (e.g. when model='auto')."""
        rec = {
            "status": "success",
            "cache_key": cache_key,
            "input_hash": input_hash,
            "output_hashes": {Path(p).name: hash_file(Path(p)) for p in outputs},
        }
        if fingerprint is not None:
            rec["fingerprint"] = fingerprint
        if models is not None:
            rec["models"] = summarize_models(models)
        if extra:
            rec.update(extra)
        self.put(stage, unit, rec)

    def record_failure(self, stage: str, unit: str, cache_key: str, input_hash: str, error: str) -> None:
        self.put(stage, unit, {"status": "failed", "cache_key": cache_key,
                               "input_hash": input_hash, "error": error})


def item_id(stage: str, unit: str, input_version: str) -> str:
    return f"{stage}:{unit}:{(input_version or 'none')[:16]}"


class RunReport:
    """Completeness of every expected unit per stage (not just units selected by --only)."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self._lock = threading.Lock()
        self.data: dict = read_json(ws.run_file, {}) or {}
        self.data.setdefault("stages", {})

    def start_stage(self, stage: str, expected_units: Iterable[str]) -> None:
        """Reset the stage so that it lists exactly the expected units, initially missing."""
        with self._lock:
            self.data["stages"][stage] = {
                "units": {u: {"status": "skipped", "reason": "not_run", "input_version": None}
                          for u in expected_units}
            }

    def set_unit(self, stage: str, unit: str, status: str, reason: str = "",
                 input_version: Optional[str] = None, error: Optional[str] = None) -> None:
        assert status in ("success", "failed", "skipped")
        with self._lock:
            st = self.data["stages"].setdefault(stage, {"units": {}})
            st["units"][unit] = {"status": status, "reason": reason,
                                 "input_version": input_version, "error": error}

    def finish_stage(self, stage: str) -> str:
        with self._lock:
            st = self.data["stages"].get(stage, {"units": {}})
            ok = all(_unit_ok(u) for u in st["units"].values())
            st["status"] = "complete" if ok else "partial"
        self.save()
        return st["status"]

    def stage_status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "missing")

    def save(self) -> None:
        with self._lock:
            write_json(self.ws.run_file, self.data)

    def failures(self, stages: Iterable[str] = UPSTREAM_STAGES) -> list[dict]:
        out = []
        for stage in stages:
            for unit, u in self.data["stages"].get(stage, {}).get("units", {}).items():
                if not _unit_ok(u):
                    out.append({"item_id": item_id(stage, unit, u.get("input_version") or ""),
                                "stage": stage, "unit": unit, "status": u["status"],
                                "reason": u.get("reason", ""), "error": u.get("error")})
        return out

    def completeness_summary(self) -> dict:
        """Upstream expected units, input versions and states; excludes times and cache-hit flags."""
        summary = {}
        for stage in UPSTREAM_STAGES:
            st = self.data["stages"].get(stage)
            if st is None:
                summary[stage] = None
                continue
            summary[stage] = {
                unit: {"state": "ok" if _unit_ok(u) else u["status"] if u["status"] == "failed" else "missing",
                       "input_version": u.get("input_version")}
                for unit, u in sorted(st["units"].items())
            }
        return summary

    def completeness_hash(self) -> str:
        return hash_obj(self.completeness_summary())

    def artifact_completeness(self, stages: Iterable[str] = UPSTREAM_STAGES) -> dict:
        """Status block embedded in aggregated artifacts so gaps travel with the data."""
        stages = tuple(stages)
        missing = [f["item_id"] for f in self.failures(stages)]
        not_run = [s for s in stages if s not in self.data["stages"]]
        return {"status": "partial" if missing or not_run else "complete",
                "missing": missing, "stages_not_run": not_run}

    def is_complete(self) -> bool:
        return not self.failures() and all(
            self.data["stages"].get(s) is not None for s in UPSTREAM_STAGES)


def _unit_ok(u: dict) -> bool:
    return u["status"] == "success" or (u["status"] == "skipped" and u.get("reason") == "cache_hit")


_tls = threading.local()


class RecordingLLM:
    """Forwards requests and records requested/actual models for the active unit."""

    def __init__(self, inner):
        self.inner = inner

    def structured(self, *args, **kwargs):
        data, meta = self.inner.structured(*args, **kwargs)
        sink = getattr(_tls, "sink", None)
        if sink is not None:
            sink.append({"requested_model": meta.get("requested_model"),
                         "actual_model": meta.get("actual_model")})
        return data, meta


@contextmanager
def model_usage():
    """Collect the models used by requests made in this thread (one unit of work)."""
    prev = getattr(_tls, "sink", None)
    _tls.sink = []
    try:
        yield _tls.sink
    finally:
        _tls.sink = prev


def current_usage_sink():
    return getattr(_tls, "sink", None)


@contextmanager
def usage_sink(sink):
    """Attach a worker thread to the unit's sink so pooled requests are recorded too."""
    prev = getattr(_tls, "sink", None)
    _tls.sink = sink
    try:
        yield
    finally:
        _tls.sink = prev


def record_usage(models: list[dict]) -> None:
    """Re-record provenance of results reused from an inner cache."""
    sink = getattr(_tls, "sink", None)
    if sink is not None:
        sink.extend(models)


def summarize_models(usage: list[dict]) -> list[dict]:
    out = []
    for u in usage:
        if u not in out:
            out.append(u)
    return out


@dataclass
class Context:
    ws: Workspace
    settings: Settings
    llm_factory: Callable[[], Any] = None
    embedder_factory: Callable[[], Any] = None
    tools_factory: Callable[[], Any] = None
    _llm: Any = field(default=None, repr=False)
    _embedder: Any = field(default=None, repr=False)
    _tools: Any = field(default=None, repr=False)

    @property
    def tools(self):
        """External media programs (ffmpeg, yt-dlp, faster-whisper)."""
        if self._tools is None:
            if self.tools_factory is None:
                from .media import MediaTools
                self._tools = MediaTools(self.settings.whisper_model)
            else:
                self._tools = self.tools_factory()
        return self._tools

    @property
    def llm(self):
        if self._llm is None:
            self._llm = RecordingLLM(self.llm_factory())
        return self._llm

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = self.embedder_factory()
        return self._embedder

    def stages(self) -> StageStore:
        return StageStore(self.ws)

    def report(self) -> RunReport:
        return RunReport(self.ws)
