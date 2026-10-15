"""Acceptance against a fixed, human-labelled baseline (spec chapter 15).

Automatable checks: pinned source versions, required concepts present and taught (or excluded only
when the baseline allows it), formulas and values preserved in cards and slides, known conflicts
recorded with both sides. Human checks: a per-slide checklist (faithful to evidence, explains the
point, fits the objective) recorded in a review file. Hours: a time log comparing manual and
tool-assisted preparation; tool-assisted human time must be at most 50% of manual."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import yaml

from .brief import BriefError, verify_approval
from .hashing import hash_obj
from .io_utils import read_json, write_json, write_text
from .state import Context

REVIEW_FIELDS = ("faithful_to_evidence", "explains_point", "fits_objective")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def _card_text(c: dict) -> str:
    parts = [c["concept"], c["definition"], c["intuition"]]
    parts += [e["text"] for e in c["examples"]] + [f["latex"] for f in c["formulas"]]
    parts += [m["text"] for m in c["misconceptions"]]
    return " ".join(parts)


def _slide_text(s: dict) -> str:
    return " ".join(s["content"]) + " " + s["speaker_script"] + " " + " ".join(
        v["description"] for v in s["visuals"])


def check_outputs(ctx: Context, baseline: dict) -> list[dict]:
    ws = ctx.ws
    results: list[dict] = []

    def add(check: str, ok: bool, detail: str):
        results.append({"check": check, "pass": bool(ok), "detail": detail})

    for sid, version in (baseline.get("source_versions") or {}).items():
        doc = read_json(ws.normalized / f"{sid}.json") or {}
        add(f"source_version:{sid}", doc.get("source_version") == version,
            f"expected {version[:12]}, got {str(doc.get('source_version'))[:12]}")

    cards = (read_json(ws.cards_file) or {}).get("cards", [])
    slides_doc = read_json(ws.slides_json) or {}
    slides = slides_doc.get("slides", [])
    brief = read_json(ws.brief_json) or {}
    excluded = {x["card_id"]: x["reason"] for x in brief.get("excluded_slide_cards", [])}
    add("slides_status", slides_doc.get("status") == "complete", f"slides status={slides_doc.get('status')}")

    def find_card(name: str, aliases: list[str]) -> Optional[dict]:
        keys = [_norm(x) for x in [name, *aliases]]
        for match in (lambda k, c: k == c, lambda k, c: k in c, lambda k, c: c in k):
            card = next((c for c in cards for k in keys if match(k, _norm(c["concept"]))), None)
            if card:
                return card
        return None

    def slides_for(card: dict) -> list[dict]:
        return [s for s in slides if card["card_id"] in s["card_ids"]]

    for rc in baseline.get("required_concepts", []):
        card = find_card(rc["name"], rc.get("aliases", []))
        if card is None:
            add(f"concept:{rc['name']}", False, "no card found")
            continue
        taught = slides_for(card)
        if taught:
            add(f"concept:{rc['name']}", True, f"{card['card_id']} on {', '.join(s['slide_id'] for s in taught)}")
        elif card["card_id"] in excluded and rc.get("allow_exclusion"):
            add(f"concept:{rc['name']}", True, f"excluded as allowed by baseline: {excluded[card['card_id']]}")
        else:
            add(f"concept:{rc['name']}", False,
                f"{card['card_id']} not taught" + (" (excluded, but baseline requires it)"
                                                   if card["card_id"] in excluded else ""))

    def check_fragment(kind: str, item: dict, field: str):
        card = find_card(item["concept"], item.get("aliases", []))
        frag = _norm(item[field])
        if card is None:
            add(f"{kind}:{item[field]}", False, f"concept {item['concept']} has no card")
            return
        in_card = frag in _norm(_card_text(card))
        in_slides = any(frag in _norm(_slide_text(s)) for s in slides_for(card))
        add(f"{kind}:{item[field]}", in_card and in_slides,
            f"in card {card['card_id']}: {in_card}; in its slides: {in_slides}")

    for f in baseline.get("formulas", []):
        check_fragment("formula", f, "latex")
    for v in baseline.get("values", []):
        check_fragment("value", v, "text")

    for kc in baseline.get("known_conflicts", []):
        card = find_card(kc["concept"], kc.get("aliases", []))
        ok = False
        if card:
            for conflict in card["conflicts"]:
                claims = " ".join(cl["statement"] for cl in conflict["claims"])
                if all(_norm(m) in _norm(claims) for m in kc["mentions"]) and \
                        all(cl.get("evidence_segment_ids") for cl in conflict["claims"]):
                    ok = True
        add(f"conflict:{kc['concept']}", ok, f"mentions {kc['mentions']} kept with evidence: {ok}")
    return results


REVIEWED_FIELDS = ("slide_id", "chapter_id", "title", "learning_objective", "content", "visuals",
                   "speaker_script", "card_ids", "evidence_segment_ids", "cards_version", "brief_hash",
                   "order_hash", "completeness_hash")


def slide_review_hash(slide: dict) -> str:
    """Identity of exactly what a human reviewed: the slide content plus its approval binding.
    Any edit (new script, changed Brief or cards) yields a new hash and requires re-review."""
    return hash_obj({k: slide.get(k) for k in REVIEWED_FIELDS})


def check_binding(ctx: Context) -> list[dict]:
    """The deliverable must belong to the current, still-valid approval and be the exported items."""
    results = []
    try:
        bound = verify_approval(ctx)
        approval = bound["approval"]
        results.append({"check": "approval", "pass": True, "detail": "approval valid for current inputs"})
    except BriefError as e:
        return [{"check": "approval", "pass": False, "detail": str(e)}]
    doc = read_json(ctx.ws.slides_json) or {}
    keys = ("cards_version", "brief_hash", "order_hash", "completeness_hash")
    mismatched = [k for k in keys if doc.get(k) != approval.get(k)]
    results.append({"check": "slides_bound_to_approval", "pass": not mismatched,
                    "detail": f"mismatched: {mismatched}" if mismatched else "slides.json matches approval"})
    tampered = []
    for s in doc.get("slides", []):
        item = read_json(ctx.ws.slide_items / f"{s['slide_id']}.json")
        if item is None or slide_review_hash(item) != slide_review_hash(s) or \
                any(s.get(k) != approval.get(k) for k in keys):
            tampered.append(s["slide_id"])
    results.append({"check": "slides_match_generated_items", "pass": not tampered,
                    "detail": f"differs from slides/items or other approval: {tampered}" if tampered
                    else "all slides match their validated items"})
    return results


def human_checklist(slides: list[dict], review: dict) -> tuple[str, list[dict], dict]:
    lines = ["# 逐張人工審核清單", "",
             "在 review 檔中為每張投影片記錄 " + "、".join(REVIEW_FIELDS) + "（true/false）、reviewer，"
             "以及下列 content_hash。投影片內容或核准版本改變後 hash 會變，必須重新確認。", ""]
    results, template = [], {}
    for s in slides:
        h = slide_review_hash(s)
        r = review.get(s["slide_id"], {}) or {}
        current = r.get("content_hash") == h
        done = current and all(r.get(k) is True for k in REVIEW_FIELDS) and bool(r.get("reviewer"))
        detail = ("confirmed" if done else "reviewed an older version; re-review required"
                  if r and not current else "pending or rejected")
        results.append({"check": f"human:{s['slide_id']}", "pass": done, "detail": detail})
        lines.append(f"- [{'x' if done else ' '}] {s['slide_id']} {s['title']}（content_hash: `{h}`）"
                     + ("" if done or not r or current else " ⚠ 內容已變更，需重新審核"))
        template[s["slide_id"]] = {"content_hash": h, **{k: (r.get(k) if current else None) for k in REVIEW_FIELDS},
                                   "reviewer": r.get("reviewer") if current else None,
                                   "notes": r.get("notes", "") if current else ""}
    return "\n".join(lines) + "\n", results, template


def hours(timelog: Path) -> dict:
    """CSV columns: method (manual|tool), task, human_minutes, machine_wait_minutes."""
    totals = {"manual": 0.0, "tool": 0.0, "tool_machine_wait": 0.0}
    with open(timelog, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            m = row["method"].strip()
            if m not in ("manual", "tool"):
                raise ValueError(f"unknown method {m!r} in {timelog}")
            totals[m] += float(row["human_minutes"] or 0)
            if m == "tool":
                totals["tool_machine_wait"] += float(row.get("machine_wait_minutes") or 0)
    ratio = totals["tool"] / totals["manual"] if totals["manual"] else None
    return {**totals, "ratio": ratio, "pass": ratio is not None and ratio <= 0.5}


def run_acceptance(ctx: Context, baseline_path: Path, review_path: Optional[Path],
                   timelog: Optional[Path]) -> dict:
    baseline = yaml.safe_load(Path(baseline_path).read_text(encoding="utf-8")) or {}
    checks = check_binding(ctx) + check_outputs(ctx, baseline)
    slides = (read_json(ctx.ws.slides_json) or {}).get("slides", [])
    review = yaml.safe_load(Path(review_path).read_text(encoding="utf-8")) if review_path else {}
    checklist, human, template = human_checklist(slides, review or {})
    write_text(ctx.ws.p("reports", "review.template.yaml"),
               yaml.safe_dump(template, allow_unicode=True, sort_keys=False))
    time = hours(timelog) if timelog else None
    report = {
        "baseline": baseline.get("name"),
        "automated": checks, "human": human, "hours": time,
        "automated_pass": all(c["pass"] for c in checks),
        "human_pass": bool(human) and all(c["pass"] for c in human),
        "hours_pass": bool(time and time["pass"]),
    }
    report["pass"] = report["automated_pass"] and report["human_pass"] and report["hours_pass"]
    out = ctx.ws.p("reports", "acceptance.json")
    write_json(out, report)
    md = [f"# 驗收：{baseline.get('name', '')}", "",
          f"- 自動檢查：{'通過' if report['automated_pass'] else '未通過'}",
          f"- 人工逐張審核：{'通過' if report['human_pass'] else '未完成或未通過'}",
          f"- 人工總工時：{_hours_line(time)}",
          f"- 結論：{'首版驗收通過' if report['pass'] else '未通過，不能宣稱首版驗收完成'}", "",
          "## 自動檢查", ""]
    md += [f"- [{'x' if c['pass'] else ' '}] {c['check']}：{c['detail']}" for c in checks]
    write_text(ctx.ws.p("reports", "acceptance.md"), "\n".join(md) + "\n\n" + checklist)
    return report


def _hours_line(t: Optional[dict]) -> str:
    if not t:
        return "未提供時間紀錄"
    r = f"{t['ratio']:.0%}" if t["ratio"] is not None else "n/a"
    return (f"純人工 {t['manual']:.0f} 分、工具輔助 {t['tool']:.0f} 分（比例 {r}，"
            f"{'≤ 50% 通過' if t['pass'] else '未達減半'}；機器等待另計 {t['tool_machine_wait']:.0f} 分）")
