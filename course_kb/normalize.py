"""Conversion of every source into a NormalizedDoc made of Segments with locators."""
from __future__ import annotations

import re
from typing import Optional

from .hashing import hash_obj

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def make_segment(source_id: str, kind: str, locator: str, text: str, **extra) -> dict:
    seg = {"segment_id": "", "kind": kind, "locator": locator, "text": text, **extra}
    digest = hash_obj({"source_id": source_id, "kind": kind, "locator": locator, "text": text})
    seg["segment_id"] = f"{source_id}:{digest}"
    return seg


def ensure_unique(segments: list[dict]) -> None:
    seen = set()
    for s in segments:
        if s["segment_id"] in seen:
            raise ValueError(f"duplicate segment_id {s['segment_id']} at {s['locator']}")
        seen.add(s["segment_id"])


def parse_markdown(source_id: str, text: str) -> list[dict]:
    """Headings become 'heading' segments; paragraphs become 'text' segments located as
    'H1 > H2 ¶n'. Lines starting with '#' inside fenced code are not headings."""
    segments: list[dict] = []
    path: list[tuple[int, str]] = []
    para: list[str] = []
    para_no = 0
    in_fence = False

    def heading_path() -> str:
        return " > ".join(t for _, t in path)

    def flush():
        nonlocal para, para_no
        body = "\n".join(para).strip("\n")
        if body.strip():
            para_no += 1
            hp = heading_path()
            loc = f"{hp} ¶{para_no}" if hp else f"¶{para_no}"
            segments.append(make_segment(source_id, "text", loc, body))
        para = []

    for line in text.replace("\r\n", "\n").split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            para.append(line)
            continue
        if in_fence:
            para.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            para_no = 0
            segments.append(make_segment(source_id, "heading", heading_path(), title, level=level))
            continue
        if not line.strip():
            flush()
            continue
        para.append(line)
    flush()
    ensure_unique(segments)
    return segments


def markdown_title(text: str) -> Optional[str]:
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            m = _HEADING.match(line)
            if m and len(m.group(1)) == 1:
                return m.group(2).strip()
    return None


# ---------------------------------------------------------------- subtitles

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _to_sec(m: re.Match) -> float:
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4).ljust(3, "0")) / 1000


def fmt_time(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_subtitles(text: str) -> list[dict]:
    """Parse SRT/WebVTT into cues, stripping tags and merging rolling duplicate lines
    typical of automatic captions."""
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip()]
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        ms = list(_TIME.finditer(lines[idx]))
        if len(ms) < 2:
            continue
        body = [_TAG.sub("", l).strip() for l in lines[idx + 1:]]
        body = [l for l in body if l]
        if body:
            cues.append({"start": _to_sec(ms[0]), "end": _to_sec(ms[1]), "lines": body})
    out = []
    last_line = None
    for cue in cues:
        new_lines = []
        for line in cue["lines"]:
            if line == last_line or (out and line in out[-1]["text"].split("\n")):
                continue
            new_lines.append(line)
        if cue["lines"]:
            last_line = cue["lines"][-1]
        if new_lines:
            out.append({"start": cue["start"], "end": cue["end"], "text": "\n".join(new_lines)})
        elif out:
            out[-1]["end"] = max(out[-1]["end"], cue["end"])
    return out


def cues_to_segments(source_id: str, cues: list[dict], origin: str, window: float = 30.0) -> list[dict]:
    """Group cues into transcript segments of at most `window` seconds."""
    segments = []
    cur: list[dict] = []

    def flush():
        if not cur:
            return
        start, end = cur[0]["start"], cur[-1]["end"]
        text = " ".join(c["text"].replace("\n", " ") for c in cur).strip()
        loc = f"{fmt_time(start)}-{fmt_time(end)}"
        segments.append(make_segment(source_id, "transcript", loc, text, start=start, end=end,
                                     text_origin=origin, raw_text=text))

    for cue in cues:
        if cur and cue["end"] - cur[0]["start"] > window:
            flush()
            cur = []
        cur.append(cue)
    flush()
    ensure_unique(segments)
    return segments


def guess_language(text: str) -> str:
    sample = text[:5000]
    cjk = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
    letters = sum(1 for c in sample if c.isalpha())
    return "zh" if letters and cjk / letters > 0.3 else "en"
