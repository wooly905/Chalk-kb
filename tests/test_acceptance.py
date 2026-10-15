import yaml

from course_kb.acceptance import run_acceptance
from course_kb.brief import approve
from course_kb.io_utils import read_json, write_json
from course_kb.slides import run_slides
from test_pipeline import plan, upstream


def _setup(ctx, project, exclude_partial_derivative=False):
    upstream(ctx, project)
    plan(ctx)
    if exclude_partial_derivative:
        cards = read_json(ctx.ws.cards_file)["cards"]
        pd = next(c["card_id"] for c in cards if c["concept"] == "偏微分")
        brief = read_json(ctx.ws.brief_json)
        brief["excluded_slide_cards"] = [{"card_id": pd, "reason": "受眾已修過微積分"}]
        write_json(ctx.ws.brief_json, brief)
    approve(ctx)
    return run_slides(ctx)


def _baseline(ctx, project, allow_exclusion=True):
    versions = {s: read_json(ctx.ws.normalized / f"{s}.json")["source_version"] for s in ("notes-a", "notes-b")}
    b = {"name": "fixture", "source_versions": versions,
         "required_concepts": [{"name": "梯度下降"}, {"name": "偏微分", "allow_exclusion": allow_exclusion}],
         "formulas": [{"concept": "梯度下降", "latex": "\\theta \\leftarrow \\theta - \\eta \\nabla L"}],
         "values": [{"concept": "梯度下降", "text": "0.1"}],
         "known_conflicts": [{"concept": "梯度下降", "mentions": ["0.1", "0.01"]}]}
    path = project / "baseline.yaml"
    path.write_text(yaml.safe_dump(b, allow_unicode=True), encoding="utf-8")
    return path


def _review(project, slides, ok=True):
    from course_kb.acceptance import slide_review_hash
    path = project / "review.yaml"
    data = {s["slide_id"]: {"content_hash": slide_review_hash(s), "faithful_to_evidence": ok,
                            "explains_point": True, "fits_objective": True, "reviewer": "T"} for s in slides}
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _timelog(project, manual, tool):
    path = project / "timelog.csv"
    path.write_text(f"method,task,human_minutes,machine_wait_minutes\nmanual,all,{manual},0\n"
                    f"tool,all,{tool},30\n", encoding="utf-8")
    return path


def test_acceptance_passes_only_with_all_three_parts(ctx, project):
    doc = _setup(ctx, project, exclude_partial_derivative=True)
    baseline = _baseline(ctx, project)
    rep = run_acceptance(ctx, baseline, None, None)
    assert rep["automated_pass"], rep["automated"]
    assert not rep["human_pass"] and not rep["hours_pass"] and not rep["pass"]
    rep = run_acceptance(ctx, baseline, _review(project, doc["slides"]), _timelog(project, 360, 140))
    assert rep["pass"] and rep["hours"]["ratio"] < 0.5 and rep["hours"]["tool_machine_wait"] == 30
    assert (ctx.ws.p("reports", "acceptance.md")).exists()


def test_acceptance_fails_on_excluded_required_concept_rejected_review_and_slow_tool(ctx, project):
    doc = _setup(ctx, project, exclude_partial_derivative=True)
    rep = run_acceptance(ctx, _baseline(ctx, project, allow_exclusion=False),
                         _review(project, doc["slides"], ok=False), _timelog(project, 100, 60))
    failed = {c["check"] for c in rep["automated"] if not c["pass"]}
    assert failed == {"concept:偏微分"}
    assert not rep["human_pass"] and not rep["hours_pass"] and not rep["pass"]


def test_acceptance_rejects_changed_brief_after_pass(ctx, project):
    doc = _setup(ctx, project, exclude_partial_derivative=True)
    baseline, review, log = _baseline(ctx, project), _review(project, doc["slides"]), _timelog(project, 360, 140)
    assert run_acceptance(ctx, baseline, review, log)["pass"]
    brief = read_json(ctx.ws.brief_json)
    brief["audience"] = "高中生"
    brief["objectives"] = ["別的目標"]
    write_json(ctx.ws.brief_json, brief)
    rep = run_acceptance(ctx, baseline, review, log)
    assert not rep["pass"] and not rep["automated_pass"]
    assert next(c for c in rep["automated"] if c["check"] == "approval")["pass"] is False


def test_acceptance_requires_rereview_of_edited_script(ctx, project):
    doc = _setup(ctx, project, exclude_partial_derivative=True)
    baseline, review, log = _baseline(ctx, project), _review(project, doc["slides"]), _timelog(project, 360, 140)
    assert run_acceptance(ctx, baseline, review, log)["pass"]
    slides = read_json(ctx.ws.slides_json)
    slides["slides"][0]["speaker_script"] = "從未被審核的新講稿"
    write_json(ctx.ws.slides_json, slides)
    rep = run_acceptance(ctx, baseline, review, log)
    assert not rep["pass"] and not rep["human_pass"] and not rep["automated_pass"]
    first = slides["slides"][0]["slide_id"]
    assert next(c for c in rep["human"] if c["check"] == f"human:{first}")["detail"].startswith("reviewed an older")
    assert (ctx.ws.p("reports", "review.template.yaml")).exists()
