from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

from .code_review.config import load_code_review_config
from .code_review.flow import code_review_flow
from .planning.config import load_planning_config
from .planning.flow import planning_flow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser(
        "review",
        help="Run a read-only AI review of the current Git diff.",
    )
    review.add_argument(
        "--repo",
        default=".",
        help="Repository to review; defaults to the current directory.",
    )
    review.add_argument(
        "--config",
        help="Repository configuration path; defaults to <repo>/.devflow.toml.",
    )
    review.add_argument(
        "--global-config",
        help="Global configuration path; defaults to ~/.config/devflow/config.toml.",
    )
    review.add_argument(
        "--provider",
        choices=("ollama", "openrouter"),
        help="Override the configured model provider for this run.",
    )
    review.add_argument(
        "--model",
        help="Override the configured model name for this run.",
    )
    review.add_argument(
        "--open",
        action="store_true",
        dest="open_report",
        help="Open review.md after the run.",
    )
    plan = subparsers.add_parser(
        "plan",
        help="Create a read-only implementation plan for a development request.",
    )
    plan.add_argument("request", help="Development outcome to plan.")
    plan.add_argument("--repo", default=".", help="Repository to plan against.")
    plan.add_argument("--config", help="Repository configuration path.")
    plan.add_argument("--global-config", help="Global configuration path.")
    plan.add_argument("--provider", choices=("ollama", "openrouter"))
    plan.add_argument("--model", help="Override the configured model name.")
    plan.add_argument(
        "--open",
        action="store_true",
        dest="open_report",
        help="Open plan.md after the run.",
    )
    return parser


def _open_file(path: str) -> None:
    resolved = Path(path).resolve()
    try:
        subprocess.Popen(["xdg-open", str(resolved)])
    except OSError:
        webbrowser.open(resolved.as_uri())


def _print_resolved_config(config) -> None:
    print("Resolved configuration")
    print(f"  Provider: {config.model.provider}")
    print(f"  Model: {config.model.model}")
    print(f"  Endpoint: {config.model.base_url}")
    print(f"  Repository: {config.repo_path}")
    if hasattr(config, "base_ref"):
        print(f"  Base ref: {config.base_ref}")
    print(
        "  Config files: "
        + (" + ".join(config.config_sources) if config.config_sources else "built-in defaults")
    )
    print()


def _print_summary(result: dict) -> None:
    assessment = result["assessment"]
    findings = assessment.get("findings", [])
    counts = {
        severity: sum(
            1 for finding in findings if finding["severity"] == severity
        )
        for severity in ("high", "medium", "low")
    }

    print()
    print(f"CODE REVIEW: {assessment['verdict'].upper()}")
    print(f"Score: {assessment['score']}/100")
    print(f"Confidence: {assessment['confidence']}")
    print(
        "Findings: "
        f"high={counts['high']} "
        f"medium={counts['medium']} "
        f"low={counts['low']}"
    )
    print(f"Report: {result['paths']['markdown']}")


def _run_review(args: argparse.Namespace) -> int:
    config = load_code_review_config(
        args.repo,
        args.config,
        global_config_path=args.global_config,
        provider_override=args.provider,
        model_override=args.model,
    )
    _print_resolved_config(config)

    result = code_review_flow(config)
    _print_summary(result)

    if args.open_report:
        _open_file(result["paths"]["markdown"])

    verdict = result["assessment"]["verdict"]
    return 1 if verdict == "fail" else 2 if verdict == "inconclusive" else 0


def _run_plan(args: argparse.Namespace) -> int:
    config = load_planning_config(
        args.repo,
        args.config,
        global_config_path=args.global_config,
        provider_override=args.provider,
        model_override=args.model,
    )
    _print_resolved_config(config)
    result = planning_flow(args.request, config)
    plan = result["plan"]
    print()
    print(f"DEVELOPMENT PLAN: {plan['status'].upper()}")
    print(f"Objective: {plan['objective']}")
    print(f"Report: {result['paths']['markdown']}")
    if args.open_report:
        _open_file(result["paths"]["markdown"])
    return 0 if plan["status"] == "ready" else 2


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        status = (
            _run_review(args)
            if args.command == "review"
            else _run_plan(args)
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
        return

    sys.exit(status)


if __name__ == "__main__":
    main()
