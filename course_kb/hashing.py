"""Canonical serialization and hashing used for versions and cache keys."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """UTF-8, sorted keys, fixed separators; array order is preserved."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_json_file(path: Path) -> str:
    """Semantic hash of a JSON file: indentation and key order do not matter."""
    with open(path, "r", encoding="utf-8") as f:
        return hash_obj(json.load(f))
