"""Command-line entry point."""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .config import Settings, load_env_file
from .coordinator import Coordinator
from .models import RunOptions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research-gap-agent",
                                description="Discover evidence-backed research gaps for a topic.")
    p.add_argument("topic", nargs="?", help="Research topic (required unless --resume)")
    p.add_argument("--discipline")
    p.add_argument("--from-year", type=int)
    p.add_argument("--to-year", type=int)
    p.add_argument("--methodology", help="Preferred research methodology")
    p.add_argument("--priority", choices=["theoretical", "experimental", "applied", "interdisciplinary"])
    p.add_argument("--goal", choices=["research", "tool"], default="research",
                   help="research: a study/method/theory; tool: a buildable technical artefact (also searches GitHub)")
    p.add_argument("--existing-work", default=None,
                   help="What you already built or want to differentiate from; text, or @path to a file such as a README")
    p.add_argument("--sources", default="openalex,semantic_scholar,crossref,arxiv",
                   help="Comma-separated: openalex,semantic_scholar,crossref,arxiv,pubmed,google_scholar (needs SERPAPI_KEY),github (code repositories; GITHUB_TOKEN optional)")
    p.add_argument("--per-source-limit", type=int, default=25, help="Records per query per source")
    p.add_argument("--max-papers", type=int, default=60, help="Max papers analysed in depth")
    p.add_argument("--validation-rounds", type=int, default=2)
    p.add_argument("--max-full-text", type=int, default=20, help="Open-access full texts to fetch (0 disables)")
    p.add_argument("--provider", default=None,
                   help="LLM provider: anthropic, openai, azure, gemini, mistral, groq, together, openrouter, deepseek, xai, "
                        "ollama, lmstudio, openai-compatible (default: RGA_LLM_PROVIDER or auto-detected from which key is set)")
    p.add_argument("--model", default=None, help="Model id for the provider (default: RGA_LLM_MODEL or the provider's default)")
    p.add_argument("--base-url", default=None, help="Base URL for openai-compatible/self-hosted servers (RGA_LLM_BASE_URL)")
    p.add_argument("--screen-model", default=None,
                   help="Cheaper model for title/abstract screening (default: RGA_SCREEN_MODEL or the provider's fast model, "
                        "e.g. claude-haiku-5-5 for Anthropic; 'main' = use the main model)")
    p.add_argument("--concurrency", type=int, default=None,
                   help="How many model calls may run at once for screening, extraction and gap validation "
                        "(default: RGA_LLM_CONCURRENCY or 4; 1 = strictly sequential)")
    p.add_argument("--run-id")
    p.add_argument("--resume", metavar="RUN_ID", help="Resume an interrupted run")
    p.add_argument("--until", choices=["plan", "discover", "screen", "fulltext", "extract", "landscape", "generate_gaps", "validate",
                                       "rank", "recommend", "verify", "report"], help="Stop after this stage")
    p.add_argument("--out", type=Path, help="Copy the final report to this path")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _read_arg_text(value: str | None) -> str | None:
    if value and value.startswith("@"):
        return Path(value[1:]).read_text()
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.llm_model = args.model
        settings.model = args.model
    if args.base_url:
        settings.llm_base_url = args.base_url
    if args.screen_model:
        settings.screen_model = args.screen_model
    if args.concurrency:
        settings.llm_concurrency = args.concurrency
    from .llm import build_llm

    llm = build_llm(settings)
    model_name = getattr(llm, "model", settings.llm_model or settings.model)
    print(f"LLM provider: {settings.llm_provider} ({model_name})", file=sys.stderr)
    progress = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731
    if args.resume:
        coord = Coordinator.resume(args.resume, llm, settings=settings, progress=progress)
    else:
        if not args.topic:
            print("error: a topic is required unless --resume is given", file=sys.stderr)
            return 2
        options = RunOptions(
            topic=args.topic, discipline=args.discipline, from_year=args.from_year, to_year=args.to_year,
            methodology_preference=args.methodology, priority=args.priority,
            goal=args.goal, existing_work=_read_arg_text(args.existing_work),
            sources=[s.strip() for s in args.sources.split(",") if s.strip()],
            per_source_limit=args.per_source_limit, max_papers_to_analyze=args.max_papers,
            validation_rounds=args.validation_rounds, max_full_text=args.max_full_text,
            provider=settings.llm_provider, model=model_name,
        )
        coord = Coordinator(options, llm, settings=settings, run_id=args.run_id, progress=progress)
    coord.run(until=args.until)
    if coord.report_path:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(coord.report_path, args.out)
            print(f"Report written to {args.out}", file=sys.stderr)
        else:
            print(f"Report written to {coord.report_path}", file=sys.stderr)
    if coord.export_paths:
        print("Exports: " + ", ".join(str(p) for p in coord.export_paths.values()), file=sys.stderr)
    print(f"State saved to {coord.state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
