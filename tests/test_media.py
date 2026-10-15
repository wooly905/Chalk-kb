"""Phase 2/3: YouTube subtitle selection, Whisper fallback, proofreading and keyframes."""
import shutil
import subprocess

import numpy as np
import pytest

from conftest import vtt
from course_kb.brief import BriefError, approve
from course_kb.cards import run_cards
from course_kb.chunk import run_chunk
from course_kb.ingest import run_ingest
from course_kb.io_utils import read_json
from course_kb.media import MediaTools, candidate_tracks, select_keyframes
from course_kb.slides import run_slides
from test_pipeline import plan


def write_sources(project, body):
    (project / "av.yaml").write_text("sources:\n" + body, encoding="utf-8")
    return project / "av.yaml"


YT = "  - id: yt\n    type: youtube\n    location: https://youtu.be/abc\n    rights_confirmed: true\n"


def test_track_order_manual_then_auto_original_language_only():
    info = {"language": "en",
            "subtitles": {"en": [], "zh-Hant": [], "live_chat": []},
            "automatic_captions": {"en-orig": [], "en": [], "zh-Hant": [], "fr": []}}
    keys = [(t["key"], t["auto"]) for t in candidate_tracks(info, None)]
    assert keys == [("en", False), ("en-orig", True), ("en", True)]
    zh = {"language": "zh", "subtitles": {"zh-CN": [], "zh-TW": []}, "automatic_captions": {}}
    assert [t["key"] for t in candidate_tracks(zh, None)] == ["zh-TW", "zh-CN"]


def test_youtube_manual_subtitles_skip_transcription_and_cleaning(ctx, project):
    t = ctx.fake_tools
    t.info = {"title": "梯度下降", "language": "zh", "id": "abc", "duration": 60,
              "subtitles": {"zh-TW": []}, "automatic_captions": {"zh-orig": []}}
    t.subs[("zh-TW", False)] = vtt([(0, 5, "概念：梯度｜定義：偏導數組成的向量"), (5, 9, "下一段")])
    assert run_ingest(ctx, write_sources(project, YT)) == "complete"
    doc = read_json(ctx.ws.normalized / "yt.json")
    assert doc["subtitle_selection"] == {"kind": "manual_subtitle", "track": "zh-TW", "language": "zh"}
    assert doc["segments"][0]["locator"] == "00:00:00-00:00:09"
    assert t.count("transcribe") == 0 and t.count("yt_audio") == 0
    assert ctx.fake_llm.count("clean_transcript") == 0
    assert (ctx.ws.raw / "yt.vtt").exists()


def test_youtube_english_auto_captions_are_proofread_keeping_raw_and_time(ctx, project):
    t = ctx.fake_tools
    t.info = {"title": "GD", "language": "en", "id": "abc", "duration": 300, "subtitles": {},
              "automatic_captions": {"en-orig": [], "zh-Hant": []}}
    t.subs[("en-orig", True)] = vtt([(i * 30, i * 30 + 30, f"line {i} gradient descent") for i in range(8)])
    run_ingest(ctx, write_sources(project, YT))
    doc = read_json(ctx.ws.normalized / "yt.json")
    assert doc["text_origin"] == "auto_subtitle" and doc["language"] == "en"
    segs = doc["segments"]
    assert all(s["text"].endswith("。") and not s["raw_text"].endswith("。") for s in segs)
    assert segs[0]["start"] == 0 and segs[-1]["end"] == 240
    payloads = [p for n, p in ctx.fake_llm.calls if n == "clean_transcript"]
    assert len(payloads) == 3  # 240 s in 90 s windows
    assert set(ctx.fake_llm.roles) == {"fast"}
    assert t.count("transcribe") == 0


def test_youtube_empty_subtitles_fall_back_to_whisper(ctx, project):
    t = ctx.fake_tools
    t.info = {"title": "GD", "language": "zh", "id": "abc", "duration": 60,
              "subtitles": {"zh-TW": []}, "automatic_captions": {"zh-orig": []}}
    t.subs[("zh-TW", False)] = "WEBVTT\n\n"
    t.subs[("zh-orig", True)] = vtt([(3, 3, "零長度")])
    run_ingest(ctx, write_sources(project, YT))
    doc = read_json(ctx.ws.normalized / "yt.json")
    assert doc["subtitle_selection"]["kind"] == "whisper" and doc["text_origin"] == "whisper"
    assert ("transcribe", "zh") in t.calls
    assert (ctx.ws.raw / "yt.transcript.json").exists() and (ctx.ws.raw / "yt.wav").exists()
    # Re-running reuses the cached transcription and reports a cache hit.
    run_ingest(ctx, write_sources(project, YT))
    assert t.count("transcribe") == 1
    assert read_json(ctx.ws.run_file)["stages"]["ingest"]["units"]["yt"]["reason"] == "cache_hit"


def test_whisper_rejects_english_only_models(ctx, project):
    (project / "materials" / "v.mp4").write_bytes(b"video")
    ctx.settings.whisper_model = "small.en"
    run_ingest(ctx, write_sources(project, "  - {id: v, type: media, location: materials/v.mp4}\n"))
    u = read_json(ctx.ws.run_file)["stages"]["ingest"]["units"]["v"]
    assert u["status"] == "failed" and "multilingual" in u["error"]


MEDIA_KF = "  - id: lec\n    type: media\n    location: materials/lec.mp4\n    keyframes: true\n"


def _media_with_frames(ctx, project):
    (project / "materials" / "lec.mp4").write_bytes(b"video-v1")
    t = ctx.fake_tools
    base = np.zeros((64, 64))
    with_formula = base.copy()
    with_formula[0:8, 0:8] = 80  # a new formula in one corner: small mean change, big block change
    t.thumbs = {0.0: base, 30.0: base.copy(), 45.0: with_formula, 60.0: np.full((64, 64), 120.0)}
    ctx.fake_llm.speaker_only_times.add("00:01:00")
    return write_sources(project, MEDIA_KF)


def test_media_keyframes_sampling_dedupe_vision_and_interleave(ctx, project):
    src = _media_with_frames(ctx, project)
    assert run_ingest(ctx, src) == "complete"
    manifest = read_json(ctx.ws.raw / "lec_frames" / "frames.json")
    assert manifest["candidates"] == 4  # 0, 30, 60 by interval + scene change at 45
    assert [f["start"] for f in manifest["frames"]] == [0.0, 45.0, 60.0]  # 30 s duplicate dropped
    doc = read_json(ctx.ws.normalized / "lec.json")
    frames = [s for s in doc["segments"] if s["kind"] == "frame"]
    assert [f["locator"] for f in frames] == ["00:00:00（畫面）", "00:00:45（畫面）"]  # 60 s = speaker only
    assert "\\theta" in frames[0]["text"] and "待確認" in frames[0]["text"]
    kinds = [(s["start"], s["kind"]) for s in doc["segments"]]
    assert kinds == sorted(kinds, key=lambda x: (x[0], x[1] != "transcript"))
    vision_roles = [r for r, (n, _) in zip(ctx.fake_llm.roles, ctx.fake_llm.calls) if n == "frame_describe"]
    assert vision_roles == ["vision"] * 3
    # Images are sent only as base64 data URLs.
    img = next(p for n, p in ctx.fake_llm.calls if n == "frame_describe")[0]["content"][1]
    assert img["type"] == "input_image" and img["image_url"].startswith("data:image/jpeg;base64,")


def test_frames_flow_into_cards_and_slides_and_are_verified(ctx, project):
    src = _media_with_frames(ctx, project)
    run_ingest(ctx, src)
    run_chunk(ctx)
    assert run_cards(ctx) == "complete"
    cards = read_json(ctx.ws.cards_file)["cards"]
    with_frames = [c for c in cards if c["visual_refs"]]
    assert with_frames and with_frames[0]["visual_refs"][0]["path"].startswith("raw/lec_frames/")
    plan(ctx)
    approve(ctx)
    doc = run_slides(ctx)
    assert doc["status"] == "complete", doc
    slide = next(s for s in doc["slides"] if s["visual_refs"])
    assert slide["visual_refs"][0]["start"] in (0.0, 45.0)
    assert any(r["type"] == "frame_uncertain" for r in slide["review_items"])

    # A changed frame image on disk invalidates the approval before any request.
    frame = ctx.ws.root / slide["visual_refs"][0]["path"]
    frame.write_bytes(b"tampered")
    before = ctx.fake_llm.count()
    with pytest.raises(BriefError, match="frame"):
        run_slides(ctx)
    assert ctx.fake_llm.count() == before


def test_changing_vision_model_redoes_frames_not_transcription(ctx, project):
    src = _media_with_frames(ctx, project)
    run_ingest(ctx, src)
    assert ctx.fake_tools.count("transcribe") == 1
    n_vision = ctx.fake_llm.count("frame_describe")
    run_ingest(ctx, src)
    assert ctx.fake_llm.count("frame_describe") == n_vision
    ctx.settings.vision_model = "other-vision"
    run_ingest(ctx, src)
    assert ctx.fake_tools.count("transcribe") == 1
    assert ctx.fake_llm.count("frame_describe") == 2 * n_vision


def test_local_auto_caption_subtitle_is_proofread(ctx, project):
    (project / "materials" / "auto.vtt").write_text(vtt([(0, 10, "auto line")]), encoding="utf-8")
    run_ingest(ctx, write_sources(
        project, "  - {id: s, type: subtitle, location: materials/auto.vtt, auto_captions: true}\n"))
    seg = read_json(ctx.ws.normalized / "s.json")["segments"][0]
    assert seg["text_origin"] == "auto_subtitle" and seg["text"] == "auto line。" and seg["raw_text"] == "auto line"


def test_keyframes_rejected_for_text_sources(ctx, project):
    with pytest.raises(Exception, match="keyframes"):
        run_ingest(ctx, write_sources(project, "  - {id: a, type: text, location: materials/a.md, keyframes: true}\n"))


# ---------------------------------------------------------------- real ffmpeg integration

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not installed")


def _make_video(path, source):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", source,
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-t", "12", "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)


@needs_ffmpeg
def test_real_ffmpeg_static_video_dedupes_to_one_frame(ctx, tmp_path):
    video = tmp_path / "static.mp4"
    _make_video(video, "color=c=white:s=320x240:d=12")
    ctx.settings.keyframe_interval_sec = 3
    tools = MediaTools()
    assert 11.5 < tools.duration(video) < 12.5
    frames, _ = select_keyframes(ctx, tools, "st", video)
    assert len(frames) == 1 and (ctx.ws.root / frames[0]["path"]).exists()
    wav = tmp_path / "a.wav"
    tools.extract_audio(video, wav)
    assert wav.stat().st_size > 16000 * 2 * 10  # ~12 s of 16 kHz mono 16-bit


@needs_ffmpeg
def test_real_ffmpeg_changing_video_keeps_distinct_frames(ctx, tmp_path):
    video = tmp_path / "moving.mp4"
    # Hard cut from black to white at 6 s triggers scene detection and a distinct frame.
    _make_video(video, "color=c=black:s=320x240:d=6[a];color=c=white:s=320x240:d=6[b];[a][b]concat=n=2:v=1:a=0")
    ctx.settings.keyframe_interval_sec = 4
    tools = MediaTools()
    scenes = tools.scene_changes(video, 0.3)
    assert any(5.5 < s < 6.5 for s in scenes)
    frames, _ = select_keyframes(ctx, tools, "mv", video)
    starts = [f["start"] for f in frames]
    assert starts[0] == 0.0 and any(t >= 6.0 for t in starts) and len(frames) == 2
    assert tools.thumbnail(ctx.ws.root / frames[0]["path"]).shape == (64, 64)


# ---------------------------------------------------------------- second review regressions

def _ingest_unit(ctx, sid="lec"):
    return read_json(ctx.ws.run_file)["stages"]["ingest"]["units"][sid]


def test_corrupted_frame_is_repaired_by_normal_ingest(ctx, project):
    src = _media_with_frames(ctx, project)
    run_ingest(ctx, src)
    doc = read_json(ctx.ws.normalized / "lec.json")
    frame = next(s for s in doc["segments"] if s["kind"] == "frame")
    path = ctx.ws.root / frame["path"]
    good = path.read_bytes()
    path.write_bytes(b"corrupt")
    n_vision = ctx.fake_llm.count("frame_describe")
    assert run_ingest(ctx, src) == "complete"
    assert _ingest_unit(ctx)["reason"] != "cache_hit"
    assert path.read_bytes() == good
    assert ctx.fake_llm.count("frame_describe") == n_vision  # same image content: descriptions reused
    assert read_json(ctx.ws.normalized / "lec.json") == doc


def test_base_url_and_schema_changes_invalidate_media_cache(ctx, project, monkeypatch):
    src = _media_with_frames(ctx, project)
    run_ingest(ctx, src)
    n = ctx.fake_llm.count("frame_describe")
    run_ingest(ctx, src)
    assert _ingest_unit(ctx)["reason"] == "cache_hit" and ctx.fake_llm.count("frame_describe") == n
    ctx.settings.base_url = "https://other.example/v1/"
    run_ingest(ctx, src)
    assert _ingest_unit(ctx)["reason"] != "cache_hit"
    n2 = ctx.fake_llm.count("frame_describe")
    assert n2 == 2 * n
    import course_kb.media as media
    monkeypatch.setattr(media, "FRAME_SCHEMA", {**media.FRAME_SCHEMA, "description": "v2"})
    run_ingest(ctx, src)
    assert ctx.fake_llm.count("frame_describe") == n2 + n


def test_local_auto_caption_cache_tracks_clean_schema(ctx, project, monkeypatch):
    (project / "materials" / "auto.vtt").write_text(vtt([(0, 10, "auto line")]), encoding="utf-8")
    src = write_sources(project, "  - {id: s, type: subtitle, location: materials/auto.vtt, auto_captions: true}\n")
    run_ingest(ctx, src)
    run_ingest(ctx, src)
    assert ctx.fake_llm.count("clean_transcript") == 1
    import course_kb.media as media
    monkeypatch.setattr(media, "CLEAN_SCHEMA", {**media.CLEAN_SCHEMA, "description": "v2"})
    run_ingest(ctx, src)
    assert ctx.fake_llm.count("clean_transcript") == 2


def test_frame_failure_keeps_successful_frames_for_resume(ctx, project):
    src = _media_with_frames(ctx, project)
    ctx.settings.max_workers = 1
    ctx.fake_llm.fail_frame_times.add("00:00:45")
    assert run_ingest(ctx, src) == "partial"
    assert ctx.fake_llm.count("frame_describe") == 3
    cache = read_json(ctx.ws.raw / "lec.frame_descriptions.json")
    assert len(cache["by_hash"]) == 2
    ctx.fake_llm.fail_frame_times.clear()
    assert run_ingest(ctx, src) == "complete"
    assert ctx.fake_llm.count("frame_describe") == 4  # only the failed frame is redone


def test_proofread_failure_keeps_successful_windows(ctx, project):
    t = ctx.fake_tools
    t.info = {"title": "GD", "language": "en", "id": "abc", "duration": 300, "subtitles": {},
              "automatic_captions": {"en-orig": []}}
    t.subs[("en-orig", True)] = vtt([(i * 30, i * 30 + 30, f"line {i}") for i in range(8)])
    orig = ctx.fake_llm._clean_transcript
    state = {"n": 0}

    def flaky(p):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom")
        return orig(p)

    ctx.fake_llm._clean_transcript = flaky
    assert run_ingest(ctx, write_sources(project, YT)) == "partial"
    assert run_ingest(ctx, write_sources(project, YT)) == "complete"
    assert ctx.fake_llm.count("clean_transcript") == 2 + 2  # 1 ok + 1 failed, then failed + remaining


def test_vision_actual_model_is_recorded(ctx, project):
    ctx.fake_llm.actual_model = "router-vision"
    src = _media_with_frames(ctx, project)
    run_ingest(ctx, src)
    rec = read_json(ctx.ws.stages_file)["ingest"]["lec"]
    assert {"requested_model": "fake", "actual_model": "router-vision"} in rec["models"]


def test_force_regenerates_whisper_clean_and_vision(ctx, project):
    src = _media_with_frames(ctx, project)
    run_ingest(ctx, src)
    before = (ctx.fake_llm.count("frame_describe"), ctx.fake_llm.count("clean_transcript"),
              ctx.fake_tools.count("transcribe"))
    assert before == (3, 1, 1)
    ctx.fake_llm.actual_model = "new-route"
    run_ingest(ctx, src, force=True)
    after = (ctx.fake_llm.count("frame_describe"), ctx.fake_llm.count("clean_transcript"),
             ctx.fake_tools.count("transcribe"))
    assert after == (6, 2, 2)
    rec = read_json(ctx.ws.stages_file)["ingest"]["lec"]
    assert {m["actual_model"] for m in rec["models"]} == {"new-route"}
