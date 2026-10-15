"""Regression tests for the 2026-10-15 review findings."""
import pytest

from course_kb.brief import BriefError, approve, render, run_brief
from course_kb.cards import _item_from_card, merge_items, run_cards
from course_kb.chunk import run_chunk
from course_kb.ingest import run_ingest
from course_kb.io_utils import read_json, write_json
from course_kb.llm import InputTooLargeError
from course_kb.slides import run_slides

from test_pipeline import plan, upstream


def _edit_a(project, old, new):
    p = project / "materials" / "a.md"
    p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def test_stale_upstream_blocks_reapproval_and_slides(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    _edit_a(project, "0.1", "0.9")
    run_ingest(ctx, project / "sources.yaml")  # only ingest re-run: chunks/cards are now stale
    with pytest.raises(BriefError, match="stale upstream"):
        approve(ctx)
    before = ctx.fake_llm.count()
    with pytest.raises(BriefError):
        run_slides(ctx)
    assert ctx.fake_llm.count() == before
    with pytest.raises(BriefError, match="out of date"):
        run_brief(ctx)

    run_chunk(ctx)
    run_cards(ctx)
    plan(ctx)
    approve(ctx)
    doc = run_slides(ctx)
    assert doc["status"] == "complete"
    text = ctx.ws.slides_json.read_text(encoding="utf-8")
    assert "0.9" in text and "學習率設為 0.1" not in text


def test_changed_merge_similarity_without_rerun_is_stale(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    ctx.settings.merge_similarity = 0.95
    assert any("merge_similarity" in b for b in render(ctx)["blockers"])


def test_prerequisite_cannot_map_to_its_own_card(ctx, project):
    (project / "materials" / "b.md").write_text(
        "# B\n\n概念：梯度下降法｜定義：沿負梯度更新｜先備：梯度\n", encoding="utf-8")
    upstream(ctx, project)
    plan(ctx)
    u = read_json(ctx.ws.order_file)["unresolved_prerequisites"][0]  # 梯度 needs 偏微分
    brief = read_json(ctx.ws.brief_json)
    brief["review_decisions"] = [{"item_id": u["item_id"], "decision": "map_prerequisite",
                                  "target_card_id": u["card_id"], "reason": "self"}]
    write_json(ctx.ws.brief_json, brief)
    result = render(ctx)
    assert not result["validation"]["valid"]
    assert any("own card" in e for e in result["validation"]["errors"])
    with pytest.raises(BriefError):
        approve(ctx)
    # Mapping to a real, different card creates an effective edge and is accepted.
    cards = read_json(ctx.ws.cards_file)["cards"]
    other = next(c["card_id"] for c in cards if c["card_id"] != u["card_id"] and c["concept"] == "梯度下降")
    brief["review_decisions"][0]["target_card_id"] = other
    write_json(ctx.ws.brief_json, brief)
    result = render(ctx)
    # other -> 梯度 contradicts 梯度 -> 梯度下降, so the cycle must be resolved before approval.
    assert any("cycle" in b or "order" in b for b in result["blockers"])


def test_cards_command_propagates_upstream_partial(ctx, project):
    with open(project / "sources.yaml", "a", encoding="utf-8") as f:
        f.write("  - id: missing\n    type: text\n    location: materials/nope.md\n")
    assert run_ingest(ctx, project / "sources.yaml") == "partial"
    assert run_chunk(ctx) == "partial"
    assert run_cards(ctx) == "partial"
    for f in (ctx.ws.chunks_file, ctx.ws.raw_cards_file, ctx.ws.cards_file):
        comp = read_json(f)["completeness"]
        assert comp["status"] == "partial" and any(m.startswith("ingest:missing") for m in comp["missing"])


def test_cross_batch_integration_sees_known_conflicts(ctx):
    def card(i, d):
        return {"card_id": f"x-c{i}", "concept": "梯度下降", "definition": d, "intuition": "",
                "examples": [], "formulas": [], "misconceptions": [], "prerequisites": []}
    items = [_item_from_card(card(1, "學習率 0.1")), _item_from_card(card(2, "學習率 0.01")),
             _item_from_card(card(3, "學習率 0.1"))]
    merged = merge_items(ctx, items, budget=1)
    calls = [p for n, p in ctx.fake_llm.calls if n == "merge_cards"]
    assert len(calls) == 2
    assert calls[1]["cards"][0]["known_conflicts"][0]["claims"] == ["學習率 0.1", "學習率 0.01"]
    assert merged["members"] == ["x-c1", "x-c2", "x-c3"]
    assert merged["conflicts"][0]["claims"][1]["card_ids"] == ["x-c2"]


def test_outline_respects_budget_and_splits_on_413(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    orig = ctx.fake_llm._slide_outline

    def limited(p):
        if len(p["cards"]) > 2:
            raise InputTooLargeError("413")
        return orig(p)

    ctx.fake_llm._slide_outline = limited
    doc = run_slides(ctx)
    assert doc["status"] == "complete" and not doc["failed_outline_chapters"]
    sizes = [len(p["cards"]) for n, p in ctx.fake_llm.calls if n == "slide_outline"]
    assert sizes[0] == 3 and all(s <= 2 for s in sizes[1:])

    ctx.settings.outline_batch_tokens = 1
    ctx.fake_llm.calls.clear()
    run_slides(ctx, force=True)
    assert [len(p["cards"]) for n, p in ctx.fake_llm.calls if n == "slide_outline"] == [1, 1, 1]


def test_stage_records_keep_fingerprint_and_actual_model(ctx, project):
    ctx.fake_llm.actual_model = "router-picked-model"
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    run_slides(ctx)
    stages = read_json(ctx.ws.stages_file)
    for stage in ("cards", "merge", "slides_outline", "slides"):
        rec = next(iter(stages[stage].values()))
        assert rec["models"] == [{"requested_model": "fake", "actual_model": "router-picked-model"}], stage
        fp = rec["fingerprint"]
        assert fp["model"] == "fake" and "base_url" in fp and "prompt" in fp
        assert "api_key" not in str(fp).lower()
    assert stages["brief"]["plan"]["models"][0]["actual_model"] == "router-picked-model"
    assert stages["ingest"]["notes-a"]["fingerprint"]["processor"]
