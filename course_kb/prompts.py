"""Prompts and strict JSON Schemas. Bump a version whenever its text or schema changes."""
from __future__ import annotations

PROMPT_VERSIONS = {
    "extract": "extract-1",
    "merge": "merge-2",
    "plan": "plan-1",
    "outline": "outline-1",
    "slide": "slide-1",
    "clean": "clean-1",
    "frame": "frame-1",
}


def _arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": list(props), "properties": props}


STR = {"type": "string"}
STRS = _arr(STR)

CARD_FIELDS = ["concept", "definition", "intuition", "examples", "formulas",
               "misconceptions", "prerequisites"]

EXTRACT_SCHEMA = _obj({
    "cards": _arr(_obj({
        "concept": STR,
        "definition": STR,
        "intuition": STR,
        "examples": STRS,
        "formulas": STRS,
        "misconceptions": STRS,
        "prerequisites": STRS,
        "evidence": _arr(_obj({
            "segment_ref": STR,
            "fields": _arr({"type": "string", "enum": CARD_FIELDS}),
        })),
    })),
})


def extract_instructions(lang: str) -> str:
    return f"""你是課程知識整理助理。從輸入區塊抽出零到多張「知識卡片」，每張卡片代表一個可教的概念。
規則：
- 只根據區塊內容，不加入外部知識，不自行編造例子、數值或公式。
- 略過寒暄、課務公告與離題內容；沒有可教概念時回傳空陣列。
- definition 忠於原文；intuition 以白話說明為什麼、怎麼理解；examples 只放原文中的具體例子；
  formulas 以 LaTeX 表示；misconceptions 只放原文提到或明確暗示的誤解；prerequisites 列出需要先懂的概念名稱。
- evidence 列出支持此卡片的 segment_ref（只能使用輸入中出現的 ref，例如 S1），並以 fields 標明它支持哪些欄位；至少一筆。
- 輸出語言：{lang}；保留英文專有名詞與程式碼。"""


MERGE_SCHEMA = _obj({
    "is_same_concept": {"type": "boolean"},
    "concept": STR,
    "definition": STR,
    "definition_from": STRS,
    "intuition": STR,
    "intuition_from": STRS,
    "examples": _arr(_obj({"text": STR, "from": STRS})),
    "formulas": _arr(_obj({"latex": STR, "from": STRS})),
    "misconceptions": _arr(_obj({"text": STR, "from": STRS})),
    "prerequisites": STRS,
    "conflicts": _arr(_obj({
        "topic": STR,
        "claims": _arr(_obj({"statement": STR, "from": STRS})),
    })),
})


def merge_instructions(lang: str) -> str:
    return f"""你會收到多張來自不同來源、向量相似的知識卡片（以 C1、C2… 標示）。
1. 先判斷它們是否講同一個概念，填入 is_same_concept。若不是，其餘欄位填空字串或空陣列。
2. 若是同一概念：選出最精確的 definition 並以 definition_from 標明採用的卡片；intuition 同理；
   examples（最多 5 個）、formulas、misconceptions 各項以 from 標明出自哪些卡片；不得新增卡片中沒有的內容。
3. 若不同卡片的定義、數值或結論不一致，不得自行裁決，必須寫入 conflicts，分別列出各方說法與 from。
   卡片的 known_conflicts 是先前批次已記錄的矛盾說法；請將它們與其他卡片比較，發現新的不一致時同樣寫入 conflicts。
4. 只能引用輸入中出現的卡片標示。輸出語言：{lang}。"""


PLAN_SCHEMA = _obj({
    "chapters": _arr(_obj({
        "title": STR,
        "learning_objectives": STRS,
        "card_ids": STRS,
        "suggested_visuals": STRS,
    })),
})


def plan_instructions(lang: str) -> str:
    return f"""你是課程設計師。依照受眾、課程學習目標與已排序的知識卡片摘要規劃章節。
規則：
- 必須維持輸入的卡片順序；每張卡片恰好放進一個章節，不得遺漏、重複或使用不存在的 card_id。
- 建議每章 3 到 8 張卡片；卡片不足或概念適合獨立成章時可以例外。
- learning_objectives 使用可評量的動詞（例如：說明、計算、比較、應用）。
- suggested_visuals 建議適合的圖表或視覺化。
- 輸出語言：{lang}。"""


OUTLINE_SCHEMA = _obj({
    "slides": _arr(_obj({
        "title": STR,
        "learning_objective": STR,
        "card_ids": STRS,
    })),
})


def outline_instructions(lang: str) -> str:
    return f"""你要為一個章節規劃投影片大綱。
規則：
- 依卡片順序安排；每張投影片聚焦一個主要教學重點，可整合同一概念的多張卡片，內容多的概念可拆成數張。
- required_card_ids 中的每張卡片至少要出現在一張投影片的 card_ids；只能使用本章提供的 card_id。
- learning_objective 寫出學習者看完應能理解或完成的事，使用可評量的動詞。
- 輸出語言：{lang}。"""


_VISUAL = _obj({"kind": STR, "description": STR, "frame_ref": STR})

SLIDE_SCHEMA = _obj({
    "content": STRS,
    "visuals": _arr(_VISUAL),
    "speaker_script": STR,
    "evidence_refs": STRS,
    "review_items": STRS,
})

SLIDE_CONTENT_SCHEMA = _obj({
    "content": STRS,
    "visuals": _arr(_VISUAL),
    "evidence_refs": STRS,
    "review_items": STRS,
})

SLIDE_SCRIPT_SCHEMA = _obj({
    "speaker_script": STR,
    "evidence_refs": STRS,
    "review_items": STRS,
})


def slide_instructions(lang: str, part: str = "full") -> str:
    base = f"""你要產生一張教學投影片。只依據提供的卡片與證據（以 E1、E2… 標示），不編造例子、數值或圖表資料。
- content：畫面上的精簡重點、原文例子與 LaTeX 公式。
- visuals：視覺素材與安排（圖表重繪所需的標籤、關係與強調重點）；引用影格時 frame_ref 填 F1 等標示，否則填空字串。
- speaker_script：可直接朗讀的完整口語講稿，解釋畫面重點、帶入例子與公式，並銜接下一張；不是重複條列。
- evidence_refs：支持本張內容的證據標示，至少一個，只能使用輸入中的標示。
- review_items：來源矛盾、看不清楚的畫面、資訊不足或需人工確認的地方。
- 輸出語言：{lang}；保留英文專有名詞與程式碼。"""
    if part == "content":
        return base + "\n本次只產生 content、visuals、evidence_refs 與 review_items。"
    if part == "script":
        return base + "\n本次只產生 speaker_script（依據已定稿的 content）、evidence_refs 與 review_items。"
    return base


CLEAN_SCHEMA = _obj({
    "segments": _arr(_obj({"ref": STR, "text": STR})),
})


def clean_instructions() -> str:
    return """你是逐字稿校對員。輸入是一段語音辨識或自動字幕（以 T1、T2… 標示，每段有時間範圍）。
只能做以下修改：補上標點、調整分段、修正明顯的語音辨識錯誤（例如同音錯字、專有名詞拼寫）。
禁止摘要、翻譯、改寫語氣、增加或刪除內容；保留原本的語言。
每個輸入標示都必須在輸出中出現一次，且順序相同。"""


FRAME_SCHEMA = _obj({
    "skip": {"type": "boolean"},
    "text": STR,
    "formulas": STRS,
    "diagram_labels": STRS,
    "diagram_relations": STRS,
    "uncertain": STRS,
})


def frame_instructions() -> str:
    return """你會看到教學影片中的一張畫面。只描述畫面中確實出現的內容：
- text：板書或投影片上的文字。
- formulas：畫面中的公式，以 LaTeX 表示。
- diagram_labels、diagram_relations：圖表的標籤，以及標籤之間的關係（例如「x 軸：迭代次數」「損失隨迭代下降」）。
- uncertain：看不清楚的文字或公式，標記為待確認，不要自行補齊或推測。
若畫面只有講者人像、沒有板書、投影片或圖表，skip 設為 true，其餘欄位留空。"""
