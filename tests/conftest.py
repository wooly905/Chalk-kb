"""Offline fakes: a scripted Responses-style LLM and a character-bigram embedder."""
from __future__ import annotations

import json
import re
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest

from course_kb.config import Settings
from course_kb.llm import LLMError, TruncatedError
from course_kb.state import Context, Workspace

CARD_LINE = re.compile(r"概念：(?P<concept>[^｜\n]+)(?P<rest>[^\n]*)")


def _field(rest: str, name: str) -> str:
    m = re.search(name + r"：([^｜\n]*)", rest)
    return m.group(1).strip() if m else ""


class FakeEmbedder:
    """Bag of character bigrams of the text before '：' (the concept name)."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), 512))
        for i, t in enumerate(texts):
            key = t.split("：")[0].strip()
            grams = [key[j:j + 2] for j in range(max(1, len(key) - 1))]
            for g in grams:
                out[i, zlib.crc32(g.encode()) % 512] += 1.0
        return out


class FakeLLM:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.fail_chunks: set[str] = set()
        self.truncate: dict[str, int] = {}
        self.not_same: set[str] = set()
        self.bad_evidence_chunks: set[str] = set()
        self.actual_model = "fake"
        self.roles: list[str] = []
        self.speaker_only_times: set[str] = set()
        self.fail_frame_times: set[str] = set()
        self._lock = threading.Lock()

    def count(self, name=None):
        return sum(1 for n, _ in self.calls if name is None or n == name)

    def structured(self, role, name, schema, instructions, input, max_output_tokens=4000):
        assert max_output_tokens <= 4000
        payload = json.loads(input) if isinstance(input, str) else input
        with self._lock:
            self.calls.append((name, payload))
            self.roles.append(role)
            if self.truncate.get(name, 0) > 0:
                self.truncate[name] -= 1
                raise TruncatedError("fake truncation")
        handler = getattr(self, "_" + name)
        return handler(payload), {"requested_model": "fake", "actual_model": self.actual_model}

    def _clean_transcript(self, p):
        return {"segments": [{"ref": s["ref"], "text": s["text"].strip() + "。"} for s in p["segments"]]}

    def _frame_describe(self, p):
        text = p[0]["content"][0]["text"]
        if any(t in text for t in self.fail_frame_times):
            raise LLMError("fake vision failure")
        if any(t in text for t in self.speaker_only_times):
            return {"skip": True, "text": "", "formulas": [], "diagram_labels": [],
                    "diagram_relations": [], "uncertain": []}
        return {"skip": False, "text": "概念：梯度下降｜定義：板書上的梯度下降更新規則",
                "formulas": ["\\theta \\leftarrow \\theta - \\eta \\nabla L"],
                "diagram_labels": ["x 軸：迭代次數"], "diagram_relations": ["損失隨迭代下降"],
                "uncertain": ["右下角的學習率數值"]}

    def _extract_cards(self, p):
        if p["chunk_id"] in self.fail_chunks:
            raise LLMError("fake API failure")
        cards = []
        for seg in p["segments"]:
            for m in CARD_LINE.finditer(seg["text"]):
                rest = m.group("rest")
                pre = [x.strip() for x in _field(rest, "先備").split("、") if x.strip()]
                ex = [x for x in [_field(rest, "例子")] if x]
                fm = [x for x in [_field(rest, "公式")] if x]
                ref = "S999" if p["chunk_id"] in self.bad_evidence_chunks else seg["segment_ref"]
                cards.append({"concept": m.group("concept").strip(), "definition": _field(rest, "定義"),
                              "intuition": "", "examples": ex, "formulas": fm, "misconceptions": [],
                              "prerequisites": pre,
                              "evidence": [{"segment_ref": ref, "fields": ["concept", "definition"]}]})
        return {"cards": cards}

    def _merge_cards(self, p):
        cards = p["cards"]
        if any(c["concept"] in self.not_same for c in cards):
            return {"is_same_concept": False, "concept": "", "definition": "", "definition_from": [],
                    "intuition": "", "intuition_from": [], "examples": [], "formulas": [],
                    "misconceptions": [], "prerequisites": [], "conflicts": []}
        first = cards[0]
        defs = {c["definition"] for c in cards}
        conflicts = []
        if len(defs) > 1:
            conflicts.append({"topic": "定義或數值不一致",
                              "claims": [{"statement": c["definition"], "from": [c["ref"]]} for c in cards]})
        examples = [{"text": e, "from": [c["ref"]]} for c in cards for e in c["examples"]]
        prereqs = list(dict.fromkeys(x for c in cards for x in c["prerequisites"]))
        return {"is_same_concept": True, "concept": first["concept"], "definition": first["definition"],
                "definition_from": [first["ref"]], "intuition": first["intuition"],
                "intuition_from": [first["ref"]], "examples": examples,
                "formulas": [{"latex": f, "from": [c["ref"]]} for c in cards for f in c["formulas"]],
                "misconceptions": [], "prerequisites": prereqs, "conflicts": conflicts}

    def _plan_chapters(self, p):
        ids = [c["card_id"] for c in p["cards"]]
        chapters = []
        for i in range(0, len(ids), 3):
            chapters.append({"title": f"章節 {len(p['previous_chapter_titles']) + len(chapters) + 1}",
                             "learning_objectives": ["說明核心概念"], "card_ids": ids[i:i + 3],
                             "suggested_visuals": []})
        return {"chapters": chapters}

    def _slide_outline(self, p):
        return {"slides": [{"title": c["concept"], "learning_objective": f"說明{c['concept']}",
                            "card_ids": [c["card_id"]]} for c in p["cards"]
                           if c["card_id"] in p["required_card_ids"]]}

    def _slide_full(self, p):
        card = p["cards"][0]
        return {"content": [card["definition"], *card["formulas"]], "visuals": [],
                "speaker_script": f"這一張我們來談{card['concept']}。{card['definition']}。",
                "evidence_refs": card["evidence_refs"][:1], "review_items": []}

    def _slide_content(self, p):
        card = p["cards"][0]
        return {"content": [card["definition"]], "visuals": [],
                "evidence_refs": card["evidence_refs"][:1], "review_items": []}

    def _slide_script(self, p):
        card = p["cards"][0]
        return {"speaker_script": f"講稿：{card['concept']}", "evidence_refs": card["evidence_refs"][:1],
                "review_items": []}


SOURCE_A = """# 最佳化

## 梯度

概念：梯度｜定義：梯度是多變數函數偏導數組成的向量｜先備：偏微分

## 梯度下降

概念：梯度下降｜定義：梯度下降沿負梯度方向更新參數，學習率設為 0.1｜先備：梯度｜例子：線性迴歸｜公式：\\theta \\leftarrow \\theta - \\eta \\nabla L

```python
# 這行在程式碼區塊中，不是標題
x = x - lr * grad
```
"""

SOURCE_B = """# 梯度下降法講義

概念：梯度下降法｜定義：梯度下降法以負梯度方向迭代更新參數，學習率設為 0.01｜先備：梯度｜例子：邏輯迴歸

概念：偏微分｜定義：固定其他變數，只對單一變數微分
"""


@pytest.fixture
def project(tmp_path: Path):
    mat = tmp_path / "materials"
    mat.mkdir()
    (mat / "a.md").write_text(SOURCE_A, encoding="utf-8")
    (mat / "b.md").write_text(SOURCE_B, encoding="utf-8")
    (tmp_path / "sources.yaml").write_text(
        "sources:\n"
        "  - id: notes-a\n    type: text\n    location: materials/a.md\n    license: own\n"
        "  - id: notes-b\n    type: text\n    location: materials/b.md\n    license: own\n",
        encoding="utf-8")
    return tmp_path


@pytest.fixture
def ctx(project: Path):
    llm, emb, tools = FakeLLM(), FakeEmbedder(), FakeTools()
    settings = Settings(text_model="fake", fast_model="fake", vision_model="fake", max_workers=2)
    c = Context(Workspace(project / "workspace"), settings, lambda: llm, lambda: emb, lambda: tools)
    c.fake_llm, c.fake_emb, c.fake_tools = llm, emb, tools
    return c


def vtt(cues: list[tuple[float, float, str]]) -> str:
    def ts(t):
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"
    return "WEBVTT\n\n" + "\n\n".join(f"{ts(a)} --> {ts(b)}\n{text}" for a, b, text in cues) + "\n"


class FakeTools:
    """Stands in for ffmpeg / yt-dlp / faster-whisper."""

    def __init__(self):
        self.info: dict = {}
        self.subs: dict[tuple[str, bool], str] = {}
        self.transcript = {"language": "zh", "segments": [
            {"start": 0.0, "end": 20.0, "text": "今天介紹梯度下降"},
            {"start": 20.0, "end": 50.0, "text": "概念：梯度下降｜定義：沿負梯度方向更新參數"}]}
        self.duration_sec = 70.0
        self.scenes: list[float] = [45.0]
        # thumbnail per frame time; equal arrays are duplicates
        self.thumbs: dict[float, np.ndarray] = {}
        self.calls: list[tuple] = []

    def yt_info(self, url):
        self.calls.append(("yt_info", url))
        return self.info

    def yt_subtitle(self, url, key, auto, out_dir, stem):
        self.calls.append(("yt_subtitle", key, auto))
        text = self.subs.get((key, auto))
        if text is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{stem}.{key}.vtt"
        p.write_text(text, encoding="utf-8")
        return p

    def yt_audio(self, url, out_dir, stem):
        self.calls.append(("yt_audio",))
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{stem}.webm"
        p.write_bytes(b"audio")
        return p

    def yt_video(self, url, out_dir, stem):
        self.calls.append(("yt_video",))
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{stem}.mp4"
        p.write_bytes(b"video")
        return p

    def extract_audio(self, media, wav):
        self.calls.append(("extract_audio",))
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"wav:" + Path(media).read_bytes())

    def transcribe(self, wav, language):
        self.calls.append(("transcribe", language))
        return self.transcript

    def duration(self, media):
        return self.duration_sec

    def scene_changes(self, video, threshold):
        return list(self.scenes)

    def frame_at(self, video, t, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f"jpeg@{t}".encode())

    def thumbnail(self, image):
        t = float(Path(image).stem.rsplit("_", 1)[1]) / 1000
        return self.thumbs.get(t, np.full((64, 64), t % 200))

    def count(self, name):
        return sum(1 for c in self.calls if c[0] == name)
