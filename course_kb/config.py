"""Tunable parameters and environment-based connection settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LANG = "繁體中文（台灣用語）"
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
MAX_OUTPUT_TOKENS_CAP = 4000

# Parameters documented in the spec (chapter 11).
chunk_max_tokens = 1200
transcript_window_sec = 240
clean_window_sec = 90
keyframe_interval_sec = 30
scene_threshold = 0.3
merge_similarity = 0.82
prereq_similarity = 0.75
max_workers = 4


@dataclass
class Settings:
    chunk_max_tokens: int = chunk_max_tokens
    transcript_window_sec: int = transcript_window_sec
    clean_window_sec: int = clean_window_sec
    keyframe_interval_sec: int = keyframe_interval_sec
    scene_threshold: float = scene_threshold
    merge_similarity: float = merge_similarity
    prereq_similarity: float = prereq_similarity
    max_workers: int = max_workers
    # Input token budgets for batched requests; outputs are capped at 4000.
    merge_batch_tokens: int = 2500
    brief_batch_tokens: int = 2500
    outline_batch_tokens: int = 2500
    output_tokens: int = MAX_OUTPUT_TOKENS_CAP

    base_url: str = DEFAULT_BASE_URL
    text_model: str = ""
    fast_model: str = ""
    vision_model: str = ""
    lang: str = DEFAULT_LANG
    embed_model: str = DEFAULT_EMBED_MODEL
    whisper_model: str = "small"

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        text = os.getenv("COURSE_KB_MODEL", "")
        return cls(
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            text_model=text,
            fast_model=os.getenv("COURSE_KB_FAST_MODEL") or text,
            vision_model=os.getenv("COURSE_KB_VISION_MODEL") or text,
            lang=os.getenv("COURSE_KB_LANG") or DEFAULT_LANG,
            embed_model=os.getenv("COURSE_KB_EMBED_MODEL") or DEFAULT_EMBED_MODEL,
            whisper_model=os.getenv("COURSE_KB_WHISPER_MODEL") or "small",
        )

    def model_for(self, role: str) -> str:
        return {"text": self.text_model, "fast": self.fast_model, "vision": self.vision_model}[role]
