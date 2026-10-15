"""Source list (YAML) loading and validation."""
from __future__ import annotations

from pathlib import Path

import yaml

SOURCE_TYPES = ("web", "text", "subtitle", "media", "youtube")
_DEFAULTS = {"title": None, "license": "unknown", "rights_confirmed": False,
             "language": None, "clean": True, "keyframes": False, "auto_captions": False}


class SourceError(ValueError):
    pass


def load_sources(path: Path) -> list[dict]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    items = doc.get("sources")
    if not isinstance(items, list) or not items:
        raise SourceError(f"{path}: 'sources' must be a non-empty list")
    seen: set[str] = set()
    out = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise SourceError(f"sources[{i}] must be a mapping")
        for key in ("id", "type", "location"):
            if not raw.get(key):
                raise SourceError(f"sources[{i}]: '{key}' is required")
        sid = str(raw["id"])
        if sid in seen:
            raise SourceError(f"duplicate source id: {sid}")
        if not all(c.isalnum() or c in "-_." for c in sid):
            raise SourceError(f"source id must only contain letters, digits, '-', '_' or '.': {sid}")
        seen.add(sid)
        if raw["type"] not in SOURCE_TYPES:
            raise SourceError(f"{sid}: type must be one of {', '.join(SOURCE_TYPES)}")
        if raw.get("keyframes") and raw["type"] not in ("media", "youtube"):
            raise SourceError(f"{sid}: keyframes only apply to media and youtube sources")
        src = {**_DEFAULTS, **raw, "id": sid}
        loc = str(src["location"])
        if src["type"] in ("text", "subtitle", "media") and not loc.startswith(("http://", "https://")):
            p = Path(loc)
            src["location"] = str(p if p.is_absolute() else (path.parent / p).resolve())
        out.append(src)
    return out
