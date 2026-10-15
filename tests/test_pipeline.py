import json

import pytest

from course_kb.brief import BriefError, approve, render, run_brief
from course_kb.cards import run_cards
from course_kb.chunk import run_chunk
from course_kb.hashing import hash_obj
from course_kb.ingest import run_ingest
from course_kb.io_utils import read_json, write_json
from course_kb.slides import run_slides


def upstream(ctx, project, **kw):
    run_ingest(ctx, project / "sources.yaml", **kw)
    run_chunk(ctx)
    return run_cards(ctx)


def plan(ctx):
    return run_brief(ctx, title="機器學習導論：最佳化", audience="大學二年級", objectives=["說明梯度下降"])


def card_by_concept(ctx, concept):
    cards = read_json(ctx.ws.cards_file)["cards"]
    return next(c for c in cards if c["concept"] == concept)


def test_end_to_end_text_pipeline(ctx, project):
    assert upstream(ctx, project) == "complete"
    cards = read_json(ctx.ws.cards_file)["cards"]
    concepts = [c["concept"] for c in cards]
    assert len(cards) == 3 and "梯度下降" in concepts and "梯度下降法" not in concepts

    gd = card_by_concept(ctx, "梯度下降")
    assert gd["merge_status"] == "merged" and len(gd["merged_from"]) == 2
    assert {s["source_id"] for s in gd["sources"]} == {"notes-a", "notes-b"}
    # Evidence points back to the original text and locator.
    assert all(e["text"] and e["locator"] for e in gd["evidence"])
    # Conflicting learning rates are kept with both sides and their evidence.
    assert gd["conflicts"] and len(gd["conflicts"][0]["claims"]) == 2
    assert all(cl["evidence_segment_ids"] for cl in gd["conflicts"][0]["claims"])
    assert {e["text"] for e in gd["examples"]} == {"線性迴歸", "邏輯迴歸"}

    result = plan(ctx)
    assert result["validation"]["valid"], result["validation"]
    assert not result["blockers"], result["blockers"]
    order = read_json(ctx.ws.order_file)["order"]
    ids = {c["concept"]: c["card_id"] for c in cards}
    # 偏微分 (from source B) must precede 梯度, which precedes 梯度下降.
    assert order.index(ids["偏微分"]) < order.index(ids["梯度"]) < order.index(ids["梯度下降"])
    md = ctx.ws.brief_md.read_text(encoding="utf-8")
    assert "來源矛盾" in md and "notes-b" in md

    approval = approve(ctx)
    assert approval["cards_version"] == read_json(ctx.ws.cards_file)["cards_version"]
    doc = run_slides(ctx)
    assert doc["status"] == "complete", doc
    assert len(doc["slides"]) == 3 and not doc["uncovered_card_ids"]
    for s in doc["slides"]:
        assert s["content"] and s["speaker_script"] and s["speaker_script"] not in s["content"]
        assert s["evidence_segment_ids"] and s["sources"][0]["location"].endswith((".md",))
        assert s["cards_version"] == approval["cards_version"]
    gd_slide = next(s for s in doc["slides"] if gd["card_id"] in s["card_ids"])
    assert any(r["type"] == "conflict" for r in gd_slide["review_items"])
    assert ctx.ws.slides_md.exists()


def test_markdown_code_block_hash_is_not_heading(ctx, project):
    upstream(ctx, project)
    doc = read_json(ctx.ws.normalized / "notes-a.json")
    headings = [s["text"] for s in doc["segments"] if s["kind"] == "heading"]
    assert headings == ["最佳化", "梯度", "梯度下降"]
    loc = [s["locator"] for s in doc["segments"] if s["kind"] == "text"]
    assert "最佳化 > 梯度下降 ¶2" in loc


def test_manual_edit_render_approve_no_llm_and_slides_use_new_title(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    brief = read_json(ctx.ws.brief_json)
    brief["chapters"][0]["title"] = "人工修改的章名"
    write_json(ctx.ws.brief_json, brief)
    before = ctx.fake_llm.count()
    render(ctx)
    approve(ctx)
    assert ctx.fake_llm.count() == before
    assert "人工修改的章名" in ctx.ws.brief_md.read_text(encoding="utf-8")
    run_slides(ctx)
    assert "人工修改的章名" in ctx.ws.slides_md.read_text(encoding="utf-8")
    # Re-running brief with unchanged inputs keeps manual edits.
    plan(ctx)
    assert read_json(ctx.ws.brief_json)["chapters"][0]["title"] == "人工修改的章名"


def test_changes_after_approval_stop_slides_before_any_request(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    brief = read_json(ctx.ws.brief_json)
    brief["chapters"][0]["title"] = "核准後才改"
    write_json(ctx.ws.brief_json, brief)
    before = ctx.fake_llm.count()
    with pytest.raises(BriefError, match="brief.json changed"):
        run_slides(ctx)
    assert ctx.fake_llm.count() == before

    # Tampering with the approval hash cannot bypass the check either.
    approval = read_json(ctx.ws.approval_file)
    approval["brief_hash"] = hash_obj(brief)
    write_json(ctx.ws.approval_file, approval)
    data = read_json(ctx.ws.cards_file)
    data["cards"][0]["definition"] += "（改）"
    write_json(ctx.ws.cards_file, data)
    with pytest.raises(BriefError):
        run_slides(ctx)
    assert ctx.fake_llm.count() == before


def test_stale_brief_with_manual_edits_requires_force(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    brief = read_json(ctx.ws.brief_json)
    brief["chapters"][0]["title"] = "手改"
    write_json(ctx.ws.brief_json, brief)
    with pytest.raises(BriefError, match="stale"):
        run_brief(ctx, title="新標題")
    run_brief(ctx, title="新標題", force=True)
    assert read_json(ctx.ws.brief_json)["title"] == "新標題"
    assert not ctx.ws.approval_file.exists()


def test_cards_version_ignores_formatting_but_tracks_content(ctx, project):
    upstream(ctx, project)
    data = read_json(ctx.ws.cards_file)
    v1 = data["cards_version"]
    ctx.ws.cards_file.write_text(json.dumps(data, ensure_ascii=False, indent=8, sort_keys=True), encoding="utf-8")
    plan(ctx)  # load_cards verifies the semantic version still matches
    assert read_json(ctx.ws.brief_json)["cards_version"] == v1
    (project / "materials" / "b.md").write_text(
        (project / "materials" / "b.md").read_text(encoding="utf-8").replace("0.01", "0.02"), encoding="utf-8")
    upstream(ctx, project)
    v2 = read_json(ctx.ws.cards_file)["cards_version"]
    assert v2 != v1
    assert "k001" in [c["card_id"] for c in read_json(ctx.ws.cards_file)["cards"]]
    # The old brief is bound to v1 and cannot be rendered/approved against v2.
    with pytest.raises(BriefError):
        render(ctx)


def test_failure_vs_zero_cards_partial_and_accept(ctx, project):
    (project / "materials" / "c.md").write_text("# 課務公告\n\n下週停課。\n", encoding="utf-8")
    with open(project / "sources.yaml", "a", encoding="utf-8") as f:
        f.write("  - id: notes-c\n    type: text\n    location: materials/c.md\n")
    ctx.fake_llm.fail_chunks.add("notes-b-001")
    assert upstream(ctx, project) == "partial"
    run = read_json(ctx.ws.run_file)
    units = run["stages"]["cards"]["units"]
    assert units["notes-b-001"]["status"] == "failed"
    assert units["notes-c-001"]["status"] == "success" and units["notes-c-001"]["reason"] == "0 cards"
    assert run["stages"]["cards"]["status"] == "partial"

    result = plan(ctx)
    assert any("notes-b-001" in b for b in result["blockers"])
    with pytest.raises(BriefError):
        approve(ctx)
    failure = ctx.report().failures()[0]
    brief = read_json(ctx.ws.brief_json)
    brief["review_decisions"].append({"item_id": failure["item_id"], "decision": "accept_missing",
                                      "reason": "講義 B 稍後補上"})
    # Without source B, 偏微分 is an unresolved prerequisite that also needs a decision.
    unresolved = read_json(ctx.ws.order_file)["unresolved_prerequisites"]
    assert [u["prerequisite"] for u in unresolved] == ["偏微分"]
    brief["review_decisions"].append({"item_id": unresolved[0]["item_id"], "decision": "audience_has",
                                      "reason": "受眾已修過微積分"})
    write_json(ctx.ws.brief_json, brief)
    a = approve(ctx)
    assert a["partial"] and a["accepted_failures"] == [failure["item_id"]]
    doc = run_slides(ctx)
    assert doc["status"] == "partial" and doc["missing"][0]["unit"] == "notes-b-001"

    # Fixing the failure changes completeness, so the old approval no longer holds.
    ctx.fake_llm.fail_chunks.clear()
    assert upstream(ctx, project) == "complete"
    with pytest.raises(BriefError):
        run_slides(ctx)


def test_cache_reuse_and_merge_similarity_only_redoes_merge(ctx, project):
    upstream(ctx, project)
    n_extract = ctx.fake_llm.count("extract_cards")
    n_merge = ctx.fake_llm.count("merge_cards")
    upstream(ctx, project)
    assert ctx.fake_llm.count("extract_cards") == n_extract
    assert ctx.fake_llm.count("merge_cards") == n_merge
    assert read_json(ctx.ws.run_file)["stages"]["ingest"]["units"]["notes-a"]["reason"] == "cache_hit"

    ctx.settings.merge_similarity = 0.95
    run_chunk(ctx)
    run_cards(ctx)
    assert ctx.fake_llm.count("extract_cards") == n_extract
    assert len(read_json(ctx.ws.cards_file)["cards"]) == 4  # no longer merged

    # A corrupted cache file must not be trusted.
    f = next(ctx.ws.extract_dir.glob("notes-a-*.json"))
    f.write_text("{}", encoding="utf-8")
    run_cards(ctx)
    assert ctx.fake_llm.count("extract_cards") == n_extract + 1


def test_changing_one_source_only_redoes_its_units(ctx, project):
    upstream(ctx, project)
    calls = [p["chunk_id"] for n, p in ctx.fake_llm.calls if n == "extract_cards"]
    (project / "materials" / "b.md").write_text(
        (project / "materials" / "b.md").read_text(encoding="utf-8") + "\n補充說明。\n", encoding="utf-8")
    upstream(ctx, project)
    new_calls = [p["chunk_id"] for n, p in ctx.fake_llm.calls if n == "extract_cards"][len(calls):]
    assert new_calls and all(c.startswith("notes-b") for c in new_calls)


def test_only_flag_keeps_other_sources_verified(ctx, project):
    upstream(ctx, project)
    run_ingest(ctx, project / "sources.yaml", only=["notes-a"])
    units = read_json(ctx.ws.run_file)["stages"]["ingest"]["units"]
    assert units["notes-b"]["reason"] == "cache_hit"
    (project / "materials" / "b.md").write_text("# 改了\n\n內容\n", encoding="utf-8")
    run_ingest(ctx, project / "sources.yaml", only=["notes-a"])
    units = read_json(ctx.ws.run_file)["stages"]["ingest"]["units"]
    assert units["notes-b"]["reason"] == "not_checked"
    assert ctx.report().failures(["ingest"])


def test_invalid_evidence_reference_fails_the_chunk(ctx, project):
    ctx.fake_llm.bad_evidence_chunks.add("notes-a-001")
    assert upstream(ctx, project) == "partial"
    u = read_json(ctx.ws.run_file)["stages"]["cards"]["units"]["notes-a-001"]
    assert u["status"] == "failed" and u["reason"] == "invalid_evidence"
    raw = read_json(ctx.ws.raw_cards_file)["cards"]
    assert not any(c["chunk_id"] == "notes-a-001" for c in raw)


def test_not_same_concept_keeps_cards_apart(ctx, project):
    ctx.fake_llm.not_same.add("梯度下降法")
    upstream(ctx, project)
    cards = read_json(ctx.ws.cards_file)["cards"]
    assert len(cards) == 4
    assert {c["merge_status"] for c in cards if c["concept"].startswith("梯度下降")} == {"not_same_concept"}


def test_truncation_splits_without_concatenating(ctx, project):
    ctx.fake_llm.truncate["extract_cards"] = 1
    ctx.fake_llm.truncate["slide_full"] = 1
    upstream(ctx, project)
    assert len(read_json(ctx.ws.cards_file)["cards"]) == 3
    plan(ctx)
    approve(ctx)
    doc = run_slides(ctx)
    assert doc["status"] == "complete"
    modes = sorted(s["generation_mode"] for s in doc["slides"])
    assert modes == ["full", "full", "split"]
    assert ctx.fake_llm.count("slide_content") == 1 and ctx.fake_llm.count("slide_script") == 1


def test_slides_resume_only_failed(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    orig = ctx.fake_llm._slide_full
    state = {"n": 0}

    def flaky(p):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom")
        return orig(p)

    ctx.fake_llm._slide_full = flaky
    ctx.settings.max_workers = 1
    doc = run_slides(ctx)
    assert doc["status"] == "partial" and len(doc["failed_slides"]) == 1
    before = ctx.fake_llm.count("slide_full")
    doc = run_slides(ctx)
    assert doc["status"] == "complete"
    assert ctx.fake_llm.count("slide_full") == before + 1


def test_excluded_cards_need_reason_and_are_reported(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    pd = card_by_concept(ctx, "偏微分")["card_id"]
    brief = read_json(ctx.ws.brief_json)
    brief["excluded_slide_cards"] = [{"card_id": pd, "reason": ""}]
    write_json(ctx.ws.brief_json, brief)
    assert not render(ctx)["validation"]["valid"]
    brief["excluded_slide_cards"] = [{"card_id": pd, "reason": "受眾已修過微積分"}]
    write_json(ctx.ws.brief_json, brief)
    approve(ctx)
    doc = run_slides(ctx)
    assert doc["status"] == "complete"
    assert doc["uncovered_card_ids"] == [pd] and not doc["required_uncovered_card_ids"]
    # The card is still in the Brief for traceability.
    assert pd in [c for ch in read_json(ctx.ws.brief_json)["chapters"] for c in ch["card_ids"]]


def test_outline_coverage_gap_is_repaired_in_order(ctx, project):
    upstream(ctx, project)
    plan(ctx)
    approve(ctx)
    orig = ctx.fake_llm._slide_outline
    ctx.fake_llm._slide_outline = lambda p: {"slides": orig(p)["slides"][1:]}
    doc = run_slides(ctx)
    assert doc["status"] == "complete" and not doc["uncovered_card_ids"]
    order = read_json(ctx.ws.order_file)["order"]
    assert [s["card_ids"][0] for s in doc["slides"]] == order


def test_need_material_blocks_approval(ctx, project):
    (project / "materials" / "b.md").write_text(
        "# B\n\n概念：梯度下降法｜定義：沿負梯度更新｜先備：線性代數\n", encoding="utf-8")
    upstream(ctx, project)
    result = plan(ctx)
    items = [u["item_id"] for u in read_json(ctx.ws.order_file)["unresolved_prerequisites"]]
    assert len(items) == 2 and len(result["blockers"]) == 2
    brief = read_json(ctx.ws.brief_json)
    brief["review_decisions"] = [{"item_id": i, "decision": "need_material", "reason": "缺教材"} for i in items]
    write_json(ctx.ws.brief_json, brief)
    with pytest.raises(BriefError, match="more material required"):
        approve(ctx)
