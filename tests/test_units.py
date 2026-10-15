from types import SimpleNamespace

import numpy as np
import pytest

from course_kb.chunk import chunk_document, estimate_tokens
from course_kb.embed import average_linkage_clusters
from course_kb.llm import (EmptyOutputError, ImageInputError, IncompleteError, InvalidJSONError,
                           RefusalError, SchemaValidationError, TruncatedError, parse_response,
                           validate_input)
from course_kb.normalize import cues_to_segments, parse_markdown, parse_subtitles
from course_kb.ordering import compute_order, kahn, strongly_connected

SRT = """1
00:00:01,000 --> 00:00:03,000
<i>今天介紹</i> 梯度

2
00:00:03,000 --> 00:00:05,000
今天介紹 梯度
下降法的概念

3
00:00:05,000 --> 00:00:07,000
下降法的概念
{\\an8}學習率很重要
"""

VTT = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello <c.yellow>world</c>

00:01:00.000 --> 00:01:02.000
Second
"""


def test_subtitles_strip_tags_and_merge_rolling_lines():
    cues = parse_subtitles(SRT)
    text = " ".join(c["text"] for c in cues)
    assert "<i>" not in text and "{\\an8}" not in text
    assert text.count("今天介紹 梯度") == 1 and text.count("下降法的概念") == 1
    assert "學習率很重要" in text
    vtt = parse_subtitles(VTT)
    assert vtt[0]["text"] == "Hello world" and vtt[1]["start"] == 60.0


def test_transcript_segments_have_time_locators():
    segs = cues_to_segments("lec", parse_subtitles(VTT), "manual_subtitle")
    assert segs[0]["locator"] == "00:00:00-00:00:02"
    assert segs[1]["locator"].startswith("00:01:00")
    assert all(s["segment_id"].startswith("lec:") and len(s["segment_id"]) == 4 + 64 for s in segs)


def test_segment_ids_are_stable_and_content_sensitive():
    a = parse_markdown("s", "# T\n\npara one\n\npara two\n")
    b = parse_markdown("s", "# T\n\npara one\n\npara two\n")
    c = parse_markdown("s", "# T\n\npara one!\n\npara two\n")
    assert [x["segment_id"] for x in a] == [x["segment_id"] for x in b]
    assert a[1]["segment_id"] != c[1]["segment_id"] and a[2]["segment_id"] == c[2]["segment_id"]


def test_token_estimate():
    assert estimate_tokens("梯度下降") == 4
    assert estimate_tokens("gradient descent works") == 4  # 3 words * 1.3


def _doc(segments, type_="text"):
    return {"source_id": "s", "source_version": "v", "type": type_, "segments": segments}


def test_chunk_transcript_window():
    segs = [{"segment_id": f"s:{i}", "kind": "transcript", "locator": "", "text": "字" * 10,
             "start": i * 60.0, "end": i * 60.0 + 60} for i in range(10)]
    chunks = chunk_document(_doc(segs, "subtitle"), 1200, 240)
    assert len(chunks) > 1
    assert chunks[0]["text"].startswith("[00:00:00]")
    assert chunks[0]["chunk_id"] == "s-001"


def test_chunk_heading_split_and_heading_only_dropped():
    segs = [
        {"segment_id": "a", "kind": "heading", "level": 1, "locator": "A", "text": "A"},
        {"segment_id": "b", "kind": "text", "locator": "A ¶1", "text": "字" * 50},
        {"segment_id": "c", "kind": "heading", "level": 2, "locator": "A > B", "text": "B"},
        {"segment_id": "d", "kind": "text", "locator": "A > B ¶1", "text": "字" * 50},
        {"segment_id": "e", "kind": "heading", "level": 2, "locator": "A > C", "text": "C"},
    ]
    chunks = chunk_document(_doc(segs), 100, 240)
    assert [c["segment_ids"] for c in chunks] == [["a", "b"], ["c", "d"]]


def test_average_linkage_avoids_chaining():
    # A~B and B~C are similar, A and C are not: average linkage must not chain all three.
    vecs = np.array([[1.0, 0.0], [0.8, 0.6], [0.28, 0.96]])
    clusters = average_linkage_clusters(vecs, 0.79)
    assert [0, 1, 2] not in clusters


def test_kahn_prefers_original_order():
    assert kahn(["a", "b", "c", "d"], [("c", "b")]) == ["a", "c", "b", "d"]


def test_cycle_auto_removal_is_recorded_and_deterministic():
    cards = [{"card_id": c} for c in ["k001", "k002", "k003"]]
    prereqs = {"threshold": 0.75, "unresolved_prerequisites": [], "candidate_edges": [
        {"from": "k001", "to": "k002", "similarity": 0.9},
        {"from": "k002", "to": "k003", "similarity": 0.8},
        {"from": "k003", "to": "k001", "similarity": 0.8},
    ]}
    order = compute_order(cards, "v", prereqs, [])
    assert order["cycles"] == [["k001", "k002", "k003"]]
    assert [(e["from"], e["to"]) for e in order["removed_edges"]] == [("k002", "k003")]
    assert order["order"] == ["k003", "k001", "k002"]
    # A human decision replaces the automatic removal.
    decisions = [{"item_id": "edge:k003->k001", "decision": "remove_edge", "reason": "循環"}]
    order2 = compute_order(cards, "v", prereqs, decisions)
    assert not order2["removed_edges"] and not order2["cycles"]
    assert order2["order"] == ["k001", "k002", "k003"]


def test_strongly_connected():
    assert strongly_connected(["a", "b", "c"], [("a", "b"), ("b", "a"), ("b", "c")]) == [["a", "b"]]


# ---------------------------------------------------------------- Responses API handling

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["x"],
          "properties": {"x": {"type": "integer"}}}


def _resp(status="completed", text='{"x": 1}', output=None, reason=None):
    return SimpleNamespace(status=status, output_text=text, output=output or [],
                           incomplete_details=SimpleNamespace(reason=reason) if reason else None)


def test_parse_response_ok_with_reasoning_items():
    output = [SimpleNamespace(type="reasoning"),
              SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text='{"x": 1}')])]
    assert parse_response(_resp(output=output), SCHEMA) == {"x": 1}


@pytest.mark.parametrize("resp,err", [
    (_resp(status="incomplete", reason="max_output_tokens"), TruncatedError),
    (_resp(status="incomplete", reason="content_filter"), IncompleteError),
    (_resp(text=""), EmptyOutputError),
    (_resp(text="{not json"), InvalidJSONError),
    (_resp(text='{"x": "a"}'), SchemaValidationError),
    (_resp(output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="no")])]),
     RefusalError),
])
def test_parse_response_errors(resp, err):
    with pytest.raises(err):
        parse_response(resp, SCHEMA)


def _img(url, **extra):
    return [{"role": "user", "content": [{"type": "input_text", "text": "x"},
                                         {"type": "input_image", "image_url": url, **extra}]}]


def test_image_inputs_only_accept_base64_data_urls():
    validate_input(_img("data:image/jpeg;base64,/9j/4AAQSkZJRg=="))
    for bad in ["https://example.com/a.jpg", "http://x/a.png", "file:///c:/a.png", "C:\\a.png",
                "data:image/png;base64,@@@", "data:text/plain;base64,aGVsbG8="]:
        with pytest.raises(ImageInputError):
            validate_input(_img(bad))
    with pytest.raises(ImageInputError):
        validate_input(_img("data:image/jpeg;base64,/9j/4AAQSkZJRg==", file_id="file-1"))
    with pytest.raises(ImageInputError):
        validate_input([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}])


def test_openai_llm_sends_text_format_and_rejects_bad_images(monkeypatch):
    from course_kb.config import Settings
    from course_kb.llm import OpenAILLM

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    s = Settings(base_url="https://router.example/v1/", text_model="main", fast_model="fast",
                 vision_model="vision")
    llm = OpenAILLM(s)
    sent = []

    class FakeResponses:
        def create(self, **kw):
            sent.append(kw)
            return _resp()

    llm.client = SimpleNamespace(responses=FakeResponses())
    data, meta = llm.structured("vision", "frame", SCHEMA, "inst",
                                _img("data:image/png;base64,iVBORw0KGgo="), max_output_tokens=9000)
    assert data == {"x": 1}
    kw = sent[0]
    assert kw["model"] == "vision" and kw["max_output_tokens"] == 4000 and kw["instructions"] == "inst"
    assert kw["text"]["format"] == {"type": "json_schema", "name": "frame", "schema": SCHEMA, "strict": True}
    with pytest.raises(ImageInputError):
        llm.structured("vision", "frame", SCHEMA, "inst", _img("https://example.com/a.jpg"))
    assert len(sent) == 1
    assert str(llm.client) and s.base_url == "https://router.example/v1/"
