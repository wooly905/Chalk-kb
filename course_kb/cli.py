"""Command-line interface: python -m course_kb <command>."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings
from .state import Context, Workspace

log = logging.getLogger("course_kb")

EXIT_OK, EXIT_PARTIAL, EXIT_BLOCKED = 0, 1, 2


def build_context(workspace: str) -> Context:
    settings = Settings.from_env()

    def llm_factory():
        from .llm import OpenAILLM
        return OpenAILLM(settings)

    def embedder_factory():
        from .embed import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(settings.embed_model)

    return Context(Workspace(Path(workspace)), settings, llm_factory, embedder_factory)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="course_kb", description="Build a traceable course knowledge base.")
    p.add_argument("--workspace", default="./workspace", help="output folder (default ./workspace)")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = p.add_subparsers(dest="command", required=True)

    def plan_args(sp):
        sp.add_argument("--title", help="course title")
        sp.add_argument("--audience", help="target audience")
        sp.add_argument("--objective", action="append", dest="objectives",
                        help="course learning objective (repeatable)")

    r = sub.add_parser("run", help="ingest → chunk → cards → brief, then stop for review")
    r.add_argument("sources")
    r.add_argument("--only", action="append", help="limit ingest to these source ids (repeatable)")
    r.add_argument("--force", action="store_true", help="re-run generation stages ignoring cache")
    plan_args(r)

    i = sub.add_parser("ingest", help="fetch and normalize sources")
    i.add_argument("sources")
    i.add_argument("--only", action="append")
    i.add_argument("--force", action="store_true")

    sub.add_parser("chunk", help="split normalized documents into chunks")

    c = sub.add_parser("cards", help="extract and merge knowledge cards")
    c.add_argument("--force", action="store_true")

    b = sub.add_parser("brief", help="plan, render or approve the Course Brief")
    plan_args(b)
    g = b.add_mutually_exclusive_group()
    g.add_argument("--render-only", action="store_true", help="validate brief.json and regenerate brief.md")
    g.add_argument("--approve", action="store_true", help="approve the current brief version")
    b.add_argument("--force", action="store_true", help="re-plan and overwrite brief.json")

    s = sub.add_parser("slides", help="generate per-slide plans and speaker scripts")
    s.add_argument("--force", action="store_true")

    x = sub.add_parser("export-pptx", help="export slides.json to a .pptx deck")
    x.add_argument("--out", help="output path (default ./slides.pptx in the current directory)")

    bd = sub.add_parser("build", help="one command: run → auto-approve → slides → pptx")
    bd.add_argument("sources")
    bd.add_argument("--force", action="store_true", help="re-run generation stages ignoring cache")
    bd.add_argument("--stop-before-approve", action="store_true",
                    help="stop after the Course Brief for manual review")
    bd.add_argument("--out", help="pptx output path (default ./slides.pptx in the current directory)")
    plan_args(bd)

    a = sub.add_parser("accept", help="check outputs against a fixed baseline (chapter 15)")
    a.add_argument("baseline", help="baseline YAML (see acceptance/baseline.example.yaml)")
    a.add_argument("--review", help="per-slide human review YAML")
    a.add_argument("--timelog", help="CSV time log: method,task,human_minutes,machine_wait_minutes")
    return p


def _report_brief(result: dict) -> int:
    v = result["validation"]
    for e in v["errors"]:
        log.error("brief: %s", e)
    for w in v["warnings"]:
        log.warning("brief: %s", w)
    if result["blockers"]:
        log.warning("brief: %d item(s) must be resolved before approval (see brief/brief.md)",
                    len(result["blockers"]))
    if not v["valid"]:
        return EXIT_PARTIAL
    return EXIT_PARTIAL if result["partial"] else EXIT_OK


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    if not args.verbose:
        for noisy in ("httpx", "openai", "sentence_transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    ctx = build_context(args.workspace)
    from .brief import BriefError, approve, auto_resolve, render, run_brief
    from .cards import run_cards
    from .chunk import run_chunk
    from .ingest import run_ingest
    from .llm import FatalLLMError
    from .slides import run_slides
    from .sources import SourceError, load_course

    try:
        cmd = args.command
        if cmd in ("run", "build"):
            course = load_course(Path(args.sources))
            title = args.title if args.title is not None else course["title"]
            audience = args.audience if args.audience is not None else course["audience"]
            objectives = args.objectives if args.objectives is not None else course["objectives"]
            statuses = [run_ingest(ctx, Path(args.sources), getattr(args, "only", None), args.force),
                        run_chunk(ctx), run_cards(ctx, args.force)]
            code = _report_brief(run_brief(ctx, title, audience, objectives, args.force))
            code = max(code, EXIT_PARTIAL if "partial" in statuses else EXIT_OK)
            if cmd == "run" or args.stop_before_approve:
                log.info("Course Brief ready for review: %s", ctx.ws.brief_md)
                return code
            added = auto_resolve(ctx)
            if added:
                log.warning("build: %d review item(s) decided automatically without human review "
                            "(listed in the pptx quality report)", len(added))
            approve(ctx)
            doc = run_slides(ctx, args.force)
            from .pptx_export import export_pptx
            export_pptx(ctx, Path(args.out) if args.out else None)
            return EXIT_OK if doc["status"] == "complete" else EXIT_PARTIAL
        if cmd == "export-pptx":
            from .pptx_export import export_pptx
            export_pptx(ctx, Path(args.out) if args.out else None)
            return EXIT_OK
        if cmd == "ingest":
            return EXIT_OK if run_ingest(ctx, Path(args.sources), args.only, args.force) == "complete" else EXIT_PARTIAL
        if cmd == "chunk":
            return EXIT_OK if run_chunk(ctx) == "complete" else EXIT_PARTIAL
        if cmd == "cards":
            return EXIT_OK if run_cards(ctx, args.force) == "complete" else EXIT_PARTIAL
        if cmd == "brief":
            if args.render_only:
                return _report_brief(render(ctx))
            if args.approve:
                a = approve(ctx)
                log.info("approved brief %s (cards_version %s)%s", a["brief_hash"][:12],
                         a["cards_version"][:12], " — partial, accepted gaps remain" if a["partial"] else "")
                return EXIT_OK
            return _report_brief(run_brief(ctx, args.title, args.audience, args.objectives, args.force))
        if cmd == "slides":
            doc = run_slides(ctx, args.force)
            return EXIT_OK if doc["status"] == "complete" else EXIT_PARTIAL
        if cmd == "accept":
            from .acceptance import run_acceptance
            rep = run_acceptance(ctx, Path(args.baseline), Path(args.review) if args.review else None,
                                 Path(args.timelog) if args.timelog else None)
            log.info("acceptance: automated=%s human=%s hours=%s -> %s (see reports/acceptance.md)",
                     rep["automated_pass"], rep["human_pass"], rep["hours_pass"],
                     "PASS" if rep["pass"] else "NOT PASSED")
            return EXIT_OK if rep["pass"] else EXIT_PARTIAL
    except (BriefError, SourceError, FatalLLMError, FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("%s", e)
        return EXIT_BLOCKED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
