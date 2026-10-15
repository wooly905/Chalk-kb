"""Phase 2/3 sources: local media and YouTube.

Order of text acquisition (both Chinese and English videos): original-language manual subtitles →
original-language automatic captions → multilingual Whisper transcription. Automatic captions and
transcripts are proofread in clean_window_sec groups (raw text is kept). With keyframes enabled,
frames are sampled on a fixed interval plus scene changes, de-duplicated locally, described by the
vision model and interleaved with the transcript by time.

External programs are reached only through MediaTools so the pipeline can run offline in tests."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from .hashing import hash_file, hash_obj, sha256_bytes
from .io_utils import read_json, write_bytes, write_json
from .llm import SPLITTABLE, SchemaValidationError, to_data_url
from .normalize import cues_to_segments, fmt_time, guess_language, make_segment, parse_subtitles
from .prompts import (CLEAN_SCHEMA, FRAME_SCHEMA, PROMPT_VERSIONS, clean_instructions,
                      frame_instructions)
from .state import Context, current_usage_sink, record_usage, usage_sink

log = logging.getLogger("course_kb")
TRANSCRIBE_PROCESSOR = "whisper-1"
CLEAN_PROCESSOR = "clean-1"
FRAMES_PROCESSOR = "frames-1"
VISION_PROCESSOR = "vision-1"
THUMB = 64
DUP_MEAN_DIFF = 2.0     # mean absolute grey-level difference (0-255) for "almost identical"
DUP_BLOCK_DIFF = 6.0    # any 8x8 block changing more than this (new writing, formula) is kept


class ToolMissing(RuntimeError):
    pass


class MediaTools:
    """Thin wrappers around ffmpeg/ffprobe, yt-dlp and faster-whisper."""

    def __init__(self, whisper_model: str = "small"):
        self.whisper_model = whisper_model
        self._whisper = None

    @staticmethod
    def _require(name: str) -> str:
        path = shutil.which(name)
        if not path:
            hint = "安裝 ffmpeg 並確認 ffmpeg、ffprobe 在 PATH 中" if name.startswith("ff") else \
                "安裝 yt-dlp（pip install yt-dlp）並確認在 PATH 中"
            raise ToolMissing(f"找不到 {name}：{hint}")
        return path

    def _run(self, args: list[str], binary: bool = False):
        exe = self._require(args[0])
        p = subprocess.run([exe, *args[1:]], capture_output=True, text=not binary,
                           **({} if binary else {"encoding": "utf-8", "errors": "replace"}))
        if p.returncode != 0:
            err = p.stderr if not binary else p.stderr.decode("utf-8", "replace")
            raise RuntimeError(f"{args[0]} failed ({p.returncode}): {err[-800:]}")
        return p

    # ffmpeg / ffprobe
    def duration(self, media: Path) -> float:
        out = self._run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=nw=1:nk=1", str(media)]).stdout.strip()
        return float(out)

    def extract_audio(self, media: Path, wav: Path) -> None:
        wav.parent.mkdir(parents=True, exist_ok=True)
        self._run(["ffmpeg", "-y", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(wav)])

    def frame_at(self, video: Path, t: float, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self._run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                   "-frames:v", "1", "-q:v", "3", str(out)])

    def scene_changes(self, video: Path, threshold: float) -> list[float]:
        p = self._run(["ffmpeg", "-hide_banner", "-i", str(video), "-vf",
                       f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"])
        return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)]

    def thumbnail(self, image: Path) -> np.ndarray:
        p = self._run(["ffmpeg", "-v", "error", "-i", str(image), "-vf",
                       f"scale={THUMB}:{THUMB},format=gray", "-f", "rawvideo", "-"], binary=True)
        return np.frombuffer(p.stdout, dtype=np.uint8)[:THUMB * THUMB].reshape(THUMB, THUMB).astype(float)

    # faster-whisper
    def transcribe(self, wav: Path, language: Optional[str]) -> dict:
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise ToolMissing("語音轉錄需要 faster-whisper（pip install faster-whisper）") from e
            self._whisper = WhisperModel(self.whisper_model, device="auto", compute_type="default")
        # Our WAV is already 16 kHz mono PCM (extract_audio); decoding it here avoids depending
        # on PyAV, whose API differs across versions.
        segments, info = self._whisper.transcribe(read_wav_16k(wav), language=language, vad_filter=True)
        return {"language": info.language,
                "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]}

    # yt-dlp
    def yt_info(self, url: str) -> dict:
        return json.loads(self._run(["yt-dlp", "-J", "--skip-download", "--no-warnings", url]).stdout)

    def yt_subtitle(self, url: str, key: str, auto: bool, out_dir: Path, stem: str) -> Optional[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._run(["yt-dlp", "--skip-download", "--no-warnings",
                   "--write-auto-subs" if auto else "--write-subs", "--sub-langs", key,
                   "--sub-format", "vtt/srt/best", "-o", str(out_dir / f"{stem}.%(ext)s"), url])
        found = sorted(out_dir.glob(f"{stem}*.vtt")) + sorted(out_dir.glob(f"{stem}*.srt"))
        return found[0] if found else None

    def yt_audio(self, url: str, out_dir: Path, stem: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._run(["yt-dlp", "--no-warnings", "-f", "bestaudio/best", "-o",
                   str(out_dir / f"{stem}.%(ext)s"), url])
        return _single(out_dir, stem)

    def yt_video(self, url: str, out_dir: Path, stem: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._run(["yt-dlp", "--no-warnings", "-f", "bv*[height<=720]+ba/b[height<=720]/bv*[height<=720]",
                   "--merge-output-format", "mp4", "-o", str(out_dir / f"{stem}.%(ext)s"), url])
        return _single(out_dir, stem)


def read_wav_16k(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16 kHz mono 16-bit PCM")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _single(out_dir: Path, stem: str) -> Path:
    found = [p for p in out_dir.glob(f"{stem}.*") if not p.name.endswith((".part", ".ytdl"))]
    if not found:
        raise RuntimeError(f"yt-dlp produced no file for {stem}")
    return found[0]


# ---------------------------------------------------------------- subtitle selection

_ZH_PREF = ["zh-Hant", "zh-TW", "zh-HK", "zh", "zh-Hans", "zh-CN", "zh-SG"]


def _lang_matches(key: str, lang: str) -> bool:
    base = key.split("-")[0].lower()
    return base == lang.split("-")[0].lower()


def candidate_tracks(info: dict, lang: Optional[str]) -> list[dict]:
    """Ordered subtitle candidates in the original language: manual first, then automatic.
    Automatic captions that are machine translations of another language are excluded."""
    lang = lang or info.get("language")
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    manual = {k: v for k, v in manual.items() if k != "live_chat"}
    if not lang:
        origs = [k[:-5] for k in auto if k.endswith("-orig")]
        if origs:
            lang = origs[0]
        elif len(manual) == 1:
            lang = next(iter(manual))
    if not lang:
        return []

    def order(keys: list[str]) -> list[str]:
        if lang.lower().startswith("zh"):
            return sorted(keys, key=lambda k: _ZH_PREF.index(k) if k in _ZH_PREF else len(_ZH_PREF))
        return sorted(keys, key=lambda k: (k != lang, k))

    out = [{"key": k, "auto": False, "kind": "manual_subtitle", "language": lang}
           for k in order([k for k in manual if _lang_matches(k, lang)])]
    orig = [k for k in auto if k.endswith("-orig") and _lang_matches(k[:-5], lang)]
    same = [k for k in auto if not k.endswith("-orig") and _lang_matches(k, lang)]
    out += [{"key": k, "auto": True, "kind": "auto_subtitle", "language": lang} for k in orig + order(same)]
    return out


# ---------------------------------------------------------------- proofreading

def clean_fingerprint(ctx: Context) -> dict:
    """Everything that shapes proofreading output; used by both the inner and the outer cache."""
    s = ctx.settings
    return {"prompt": PROMPT_VERSIONS["clean"], "schema": hash_obj(CLEAN_SCHEMA), "model": s.fast_model,
            "base_url": s.base_url, "window": s.clean_window_sec, "processor": CLEAN_PROCESSOR}


def vision_fingerprint(ctx: Context) -> dict:
    s = ctx.settings
    return {"prompt": PROMPT_VERSIONS["frame"], "schema": hash_obj(FRAME_SCHEMA), "model": s.vision_model,
            "base_url": s.base_url, "processor": VISION_PROCESSOR}


def frames_fingerprint(ctx: Context) -> dict:
    s = ctx.settings
    return {"interval": s.keyframe_interval_sec, "scene": s.scene_threshold, "processor": FRAMES_PROCESSOR}


def proofread(ctx: Context, sid: str, segments: list[dict], cache_path: Path,
              force: bool = False) -> tuple[list[dict], dict]:
    """Proofread transcript segments in clean_window_sec groups with the fast model.
    Raw text and time ranges are preserved; only `text` changes. Each successful window is
    persisted immediately so a later failure does not discard it."""
    s = ctx.settings
    fp = clean_fingerprint(ctx)
    cache = read_json(cache_path) or {}
    if cache.get("fingerprint") != fp:
        cache = {"fingerprint": fp, "windows": {}}
    windows: list[list[dict]] = []
    for seg in segments:
        if windows and seg["end"] - windows[-1][0]["start"] <= s.clean_window_sec:
            windows[-1].append(seg)
        else:
            windows.append([seg])
    fresh: set[str] = set()

    def clean_window(win: list[dict]) -> list[str]:
        key = hash_obj([[w["start"], w["end"], w["raw_text"]] for w in win])
        hit = cache["windows"].get(key)
        if hit is not None and (not force or key in fresh):
            record_usage(hit.get("models", []))
            return hit["texts"]
        try:
            refs = {f"T{i + 1}": w for i, w in enumerate(win)}
            payload = {"segments": [{"ref": r, "time": w["locator"], "text": w["raw_text"]} for r, w in refs.items()]}
            data, meta = ctx.llm.structured("fast", "clean_transcript", CLEAN_SCHEMA, clean_instructions(),
                                            json.dumps(payload, ensure_ascii=False), s.output_tokens)
            if [x["ref"] for x in data["segments"]] != list(refs):
                raise SchemaValidationError("proofreading must return every segment ref once, in order")
            texts = [x["text"].strip() or refs[x["ref"]]["raw_text"] for x in data["segments"]]
            models = [_model_meta(meta)]
        except SPLITTABLE:
            if len(win) <= 1:
                raise
            mid = len(win) // 2
            texts = clean_window(win[:mid]) + clean_window(win[mid:])
            models = []
        cache["windows"][key] = {"texts": texts, "models": models}
        fresh.add(key)
        write_json(cache_path, cache)
        return texts

    out = []
    for win in windows:
        for seg, text in zip(win, clean_window(win)):
            out.append({**seg, "text": text, "cleaned": True})
    return out, fp


def _model_meta(meta: dict) -> dict:
    return {"requested_model": meta.get("requested_model"), "actual_model": meta.get("actual_model")}


def transcript_segments(sid: str, cues: list[dict], origin: str) -> list[dict]:
    """segment_id is derived from the raw (pre-proofreading) text so it stays tied to the source."""
    return cues_to_segments(sid, cues, origin=origin)


# ---------------------------------------------------------------- whisper

def whisper_cues(ctx: Context, tools: MediaTools, sid: str, audio_src: Path, language: Optional[str],
                 force: bool = False) -> tuple[list[dict], str, str]:
    """Extract 16 kHz mono audio and transcribe it (cached by audio content + model)."""
    model = ctx.settings.whisper_model
    if model.endswith(".en"):
        raise ValueError("COURSE_KB_WHISPER_MODEL must be multilingual (no '.en' models)")
    wav = ctx.ws.raw / f"{sid}.wav"
    tools.extract_audio(audio_src, wav)
    audio_hash = hash_file(wav)
    tpath = ctx.ws.raw / f"{sid}.transcript.json"
    key = hash_obj({"audio": audio_hash, "model": model, "language": language, "processor": TRANSCRIBE_PROCESSOR})
    cached = read_json(tpath)
    if cached and cached.get("cache_key") == key and not force:
        result = cached["result"]
    else:
        result = tools.transcribe(wav, language)
        write_json(tpath, {"cache_key": key, "model": model, "result": result})
    cues = [{"start": float(x["start"]), "end": float(x["end"]), "text": x["text"]}
            for x in result["segments"] if x["text"].strip()]
    return cues, result.get("language") or language or "", audio_hash


# ---------------------------------------------------------------- keyframes

def _is_duplicate(a: np.ndarray, b: np.ndarray) -> bool:
    diff = np.abs(a - b)
    if diff.mean() >= DUP_MEAN_DIFF:
        return False
    blocks = diff.reshape(THUMB // 8, 8, THUMB // 8, 8).mean(axis=(1, 3))
    return blocks.max() < DUP_BLOCK_DIFF


def select_keyframes(ctx: Context, tools: MediaTools, sid: str, video: Path) -> tuple[list[dict], dict]:
    """Interval sampling + scene changes, merged by time and de-duplicated locally."""
    s = ctx.settings
    fdir = ctx.ws.raw / f"{sid}_frames"
    manifest = fdir / "frames.json"
    video_hash = hash_file(video)
    fp = {"video": video_hash, **frames_fingerprint(ctx)}
    cached = read_json(manifest)
    if cached and cached.get("fingerprint") == fp and all((ctx.ws.root / f["path"]).exists()
                                                        and hash_file(ctx.ws.root / f["path"]) == f["image_hash"]
                                                        for f in cached["frames"]):
        return cached["frames"], fp
    duration = tools.duration(video)
    times = [float(t) for t in np.arange(0, duration, s.keyframe_interval_sec)]
    times += [t for t in tools.scene_changes(video, s.scene_threshold) if t < duration]
    times.sort()
    merged: list[float] = []
    for t in times:
        if not merged or t - merged[-1] >= 0.5:
            merged.append(t)
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    kept, last = [], None
    for t in merged:
        path = fdir / f"{sid}_{int(round(t * 1000)):09d}.jpg"
        tools.frame_at(video, t, path)
        if not path.exists():
            continue
        thumb = tools.thumbnail(path)
        if last is not None and _is_duplicate(last, thumb):
            path.unlink()
            continue
        last = thumb
        kept.append({"frame_id": path.stem, "path": path.relative_to(ctx.ws.root).as_posix(),
                     "start": round(t, 3), "image_hash": hash_file(path)})
    write_json(manifest, {"fingerprint": fp, "candidates": len(merged), "frames": kept})
    log.info("%s: %d frame candidates -> %d kept", sid, len(merged), len(kept))
    return kept, fp


def describe_frames(ctx: Context, sid: str, frames: list[dict], force: bool = False) -> tuple[list[dict], dict]:
    """Vision understanding per frame, cached by image content + vision settings. Every
    successful frame is persisted immediately; failures are raised after all frames ran, so a
    re-run only redoes the missing ones."""
    s = ctx.settings
    fp = vision_fingerprint(ctx)
    cpath = ctx.ws.raw / f"{sid}.frame_descriptions.json"
    cache = read_json(cpath) or {}
    if cache.get("fingerprint") != fp:
        cache = {"fingerprint": fp, "by_hash": {}}
    lock = threading.Lock()
    sink = current_usage_sink()

    def one(frame: dict) -> dict:
        with usage_sink(sink):
            hit = cache["by_hash"].get(frame["image_hash"])
            if hit is not None and not force:
                record_usage(hit.get("models", []))
                return hit["description"]
            url = to_data_url(ctx.ws.root / frame["path"])
            content = [{"type": "input_text", "text": f"影片時間 {fmt_time(frame['start'])} 的畫面。"},
                       {"type": "input_image", "image_url": url}]
            data, meta = ctx.llm.structured("vision", "frame_describe", FRAME_SCHEMA, frame_instructions(),
                                            [{"role": "user", "content": content}], s.output_tokens)
            with lock:
                cache["by_hash"][frame["image_hash"]] = {"description": data, "models": [_model_meta(meta)]}
                write_json(cpath, cache)
            return data

    with ThreadPoolExecutor(max_workers=max(1, s.max_workers)) as pool:
        futures = [pool.submit(one, f) for f in frames]
    errors = [f.exception() for f in futures if f.exception() is not None]
    if errors:
        raise errors[0]
    return [f.result() for f in futures], fp


def frame_segments(sid: str, frames: list[dict], descriptions: list[dict]) -> list[dict]:
    out = []
    for f, d in zip(frames, descriptions):
        if d["skip"]:
            continue
        parts = []
        if d["text"].strip():
            parts.append(f"板書文字：{d['text'].strip()}")
        if d["formulas"]:
            parts.append("公式：" + "；".join(f"${x}$" for x in d["formulas"]))
        if d["diagram_labels"]:
            parts.append("圖表標籤：" + "；".join(d["diagram_labels"]))
        if d["diagram_relations"]:
            parts.append("圖表關係：" + "；".join(d["diagram_relations"]))
        if d["uncertain"]:
            parts.append("待確認：" + "；".join(d["uncertain"]))
        if not parts:
            continue
        loc = f"{fmt_time(f['start'])}（畫面）"
        seg = make_segment(sid, "frame", loc, "\n".join(parts), start=f["start"], end=f["start"],
                           frame_id=f["frame_id"], path=f["path"], image_hash=f["image_hash"],
                           formulas=d["formulas"], uncertain=d["uncertain"])
        out.append(seg)
    return out


# ---------------------------------------------------------------- source ingestion

def media_input_hash(ctx: Context, src: dict, config: dict) -> str:
    """Version of a local media source before processing: file content + every setting that
    shapes the normalized document (same fingerprints as the inner caches)."""
    s = ctx.settings
    return hash_obj({"file": hash_file(Path(src["location"])), "config": config,
                     "whisper": [s.whisper_model, TRANSCRIBE_PROCESSOR],
                     "clean": clean_fingerprint(ctx) if src.get("clean") else None,
                     "frames": [frames_fingerprint(ctx), vision_fingerprint(ctx)] if src.get("keyframes") else None})


def frame_outputs_valid(ctx: Context, doc: dict) -> bool:
    """Frames referenced by a normalized document must still exist with their recorded content."""
    for g in doc.get("segments", []):
        if g["kind"] == "frame":
            p = ctx.ws.root / g["path"]
            if not p.exists() or hash_file(p) != g["image_hash"]:
                return False
    return True


def ingest_av(ctx: Context, tools: MediaTools, src: dict, config: dict,
              source_version: Optional[str] = None, force: bool = False) -> dict:
    """source_version: precomputed for local media (see media_input_hash); YouTube versions are
    derived from the content actually retrieved (subtitle, audio, frames)."""
    sid = src["id"]
    raw = ctx.ws.raw
    language = src.get("language")
    versions: dict = {"config": config}
    selection: dict
    video: Optional[Path] = None
    title = src.get("title")

    if src["type"] == "media":
        path = Path(src["location"])
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        cues, lang, audio_hash = whisper_cues(ctx, tools, sid, path, language, force)
        versions.update(file=hash_file(path), audio=audio_hash, whisper=ctx.settings.whisper_model)
        selection = {"kind": "whisper", "language": lang}
        origin = "whisper"
        video = path
        title = title or path.stem
    else:
        if not src.get("rights_confirmed"):
            raise PermissionError("YouTube source requires rights_confirmed: true")
        info = tools.yt_info(src["location"])
        title = title or info.get("title")
        versions["video"] = {k: info.get(k) for k in ("id", "duration")}
        cues, selection, origin = [], {}, ""
        for track in candidate_tracks(info, language):
            tmp = raw / f"{sid}_subs"
            if tmp.exists():
                shutil.rmtree(tmp)
            sub = tools.yt_subtitle(src["location"], track["key"], track["auto"], tmp, sid)
            if sub is None:
                continue
            data = sub.read_bytes()
            parsed = parse_subtitles(data.decode("utf-8-sig", errors="replace"))
            parsed = [c for c in parsed if c["text"].strip() and c["end"] > c["start"]]
            if not parsed:
                log.info("%s: subtitle %s empty or invalid; trying next", sid, track["key"])
                continue
            write_bytes(raw / f"{sid}{sub.suffix}", data)
            shutil.rmtree(tmp, ignore_errors=True)
            cues, origin = parsed, track["kind"]
            selection = {"kind": track["kind"], "track": track["key"], "language": track["language"]}
            versions["subtitle"] = sha256_bytes(data)
            break
        if not cues:
            audio = tools.yt_audio(src["location"], raw / f"{sid}_dl", f"{sid}.audio")
            cues, lang, audio_hash = whisper_cues(ctx, tools, sid, audio, language or info.get("language"), force)
            versions.update(audio=audio_hash, whisper=ctx.settings.whisper_model)
            selection = {"kind": "whisper", "language": lang}
            origin = "whisper"
        if src.get("keyframes"):
            video = tools.yt_video(src["location"], raw / f"{sid}_dl", f"{sid}.video")

    segments = transcript_segments(sid, cues, origin)
    if src.get("clean") and origin in ("auto_subtitle", "whisper") and segments:
        segments, clean_fp = proofread(ctx, sid, segments, raw / f"{sid}.clean.json", force)
        versions["clean"] = clean_fp
    if src.get("keyframes"):
        frames, frames_fp = select_keyframes(ctx, tools, sid, video)
        descriptions, vision_fp = describe_frames(ctx, sid, frames, force)
        fsegs = frame_segments(sid, frames, descriptions)
        versions.update(frames=frames_fp, frame_hashes=[f["image_hash"] for f in frames], vision=vision_fp)
        segments = sorted(segments + fsegs, key=lambda g: (g["start"], g["kind"] != "transcript"))
    if not segments:
        raise RuntimeError("source produced no transcript or frame content")
    full = " ".join(g["text"] for g in segments)
    return {
        "source_id": sid, "source_version": source_version or hash_obj(versions), "type": src["type"],
        "title": title,
        "location": src["location"], "license": src.get("license"),
        "language": selection.get("language") or language or guess_language(full),
        "text_origin": origin, "subtitle_selection": selection, "segments": segments,
    }
