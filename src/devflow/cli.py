from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import webbrowser
import json
from contextlib import contextmanager
from pathlib import Path

from .code_review.config import load_code_review_config
from .code_review.flow import code_review_flow
from .code_review.tasks import _run_command
from .implementation.flow import implementation_flow
from .planning.config import load_planning_config
from .planning.artifacts import create_plan_run_dir
from .planning.flow import planning_flow
from .repository_context.config import load_serena_context_config
from .repository_context.flow import serena_context_flow
from .repository_context.serena import SerenaContextRunError


def _add_common_config_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--repo",
        default=".",
        help="Target repository; defaults to the current directory.",
    )
    command.add_argument(
        "--config",
        help="Repository configuration path; defaults to <repo>/.devflow.toml.",
    )
    command.add_argument(
        "--global-config",
        help="Global configuration path; defaults to ~/.config/devflow/config.toml.",
    )
    command.add_argument(
        "--provider",
        choices=("ollama", "openrouter"),
        help="Override the configured model provider for this run.",
    )
    command.add_argument(
        "--model",
        help="Override the configured model name for this run.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser(
        "review",
        help="Run a read-only AI review of the current Git diff.",
    )
    _add_common_config_arguments(review)
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
    _add_common_config_arguments(plan)
    plan.add_argument(
        "--context",
        help="Reuse an existing Serena context.json instead of running discovery.",
    )
    plan.add_argument(
        "--from-plan",
        help="Revise an existing plan.json rather than creating a plan from scratch.",
    )
    plan.add_argument(
        "--open",
        action="store_true",
        dest="open_report",
        help="Open plan.md after the run.",
    )
    plan.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Approve context research and replanning gates without prompting.",
    )
    plan.add_argument(
        "--open-plan",
        action="store_true",
        help="Open plan.json after the run without prompting.",
    )
    plan.add_argument(
        "--answers",
        help="JSON answers file created by a previous deferred planning run.",
    )
    plan.add_argument(
        "--context-hints",
        help="JSON research-hints file created by a stopped context run.",
    )
    plan.add_argument(
        "--human-input-ledger",
        help="Reuse reconciled human answers and active research items from a prior run.",
    )
    plan.add_argument(
        "--architecture-decisions",
        help="JSON architecture decisions file created by a stopped planning run.",
    )
    serena = subparsers.add_parser(
        "serena-context",
        help="Discover grounded repository context with Serena.",
    )
    serena.add_argument("request", help="Development outcome to investigate.")
    _add_common_config_arguments(serena)
    implement = subparsers.add_parser(
        "implement",
        help="Create and optionally apply edits from an approved implementation plan.",
    )
    implement.add_argument("plan", help="Path to an approved plan.json.")
    _add_common_config_arguments(implement)
    implement.add_argument(
        "--yes", "-y", action="store_true",
        help="Apply a valid implementation proposal without prompting.",
    )
    return parser


def _open_file(path: str) -> None:
    resolved = Path(path).resolve()
    try:
        subprocess.Popen(["xdg-open", str(resolved)])
    except OSError:
        webbrowser.open(resolved.as_uri())


class _TeeStdout:
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, value):
        self.console.write(value)
        self.log_file.write(value)
        self.log_file.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)


class _ExcludePrefectLogs(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("prefect")


@contextmanager
def _capture_plan_log(run_dir: Path):
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = _TeeStdout(original_stdout, log_file)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        root_handler = logging.FileHandler(log_path, encoding="utf-8")
        root_handler.setLevel(logging.INFO)
        root_handler.setFormatter(formatter)
        root_handler.addFilter(_ExcludePrefectLogs())
        prefect_handler = logging.FileHandler(log_path, encoding="utf-8")
        prefect_handler.setLevel(logging.INFO)
        prefect_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        prefect_logger = logging.getLogger("prefect")
        previous_root_level = root_logger.level
        previous_prefect_level = prefect_logger.level
        root_logger.setLevel(min(previous_root_level, logging.INFO))
        prefect_logger.setLevel(min(previous_prefect_level, logging.INFO))
        root_logger.addHandler(root_handler)
        prefect_logger.addHandler(prefect_handler)
        try:
            yield str(log_path.resolve())
        finally:
            prefect_logger.removeHandler(prefect_handler)
            root_logger.removeHandler(root_handler)
            prefect_logger.setLevel(previous_prefect_level)
            root_logger.setLevel(previous_root_level)
            prefect_handler.close()
            root_handler.close()
            sys.stdout = original_stdout


def _confirm_open_plan(path: str, force_open: bool) -> bool:
    if force_open:
        logging.getLogger(__name__).info("Open-plan gate auto-approved")
        return True
    if not sys.stdin.isatty():
        logging.getLogger(__name__).info(
            "Open-plan gate skipped because stdin is not interactive"
        )
        return False
    answer = input(f"Open the JSON plan {path}? [y/N]: ").strip().lower()
    approved = answer in {"y", "yes"}
    logging.getLogger(__name__).info(
        "Open-plan gate %s", "approved" if approved else "declined"
    )
    return approved


def _confirm_open_diagnostic(path: str) -> bool:
    if not sys.stdin.isatty():
        logging.getLogger(__name__).info(
            "Open-diagnostic gate skipped because stdin is not interactive"
        )
        return False
    answer = input(f"Open the Serena diagnostic {path}? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _handle_serena_error(error: SerenaContextRunError) -> int:
    print()
    print(f"SERENA CONTEXT FAILED: {error}")
    print(f"Diagnostic: {error.diagnostic_path}")
    if _confirm_open_diagnostic(error.diagnostic_path):
        _open_file(error.diagnostic_path)
    return 2


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
    serena_config = load_serena_context_config(
        args.repo,
        args.config,
        global_config_path=args.global_config,
        provider_override=args.provider,
        model_override=args.model,
    )
    run_dir = create_plan_run_dir(config.output_dir)
    with _capture_plan_log(run_dir) as log_path:
        _print_resolved_config(config)
        review_config = load_code_review_config(
            args.repo, args.config, global_config_path=args.global_config
        )
        repo = Path(config.repo_path)
        def git_value(*command):
            result = subprocess.run(
                ["git", *command], cwd=repo, capture_output=True, text=True, check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
        branch = git_value("branch", "--show-current")
        commit = git_value("rev-parse", "HEAD")
        status = git_value("status", "--short")
        repository_state = {
            "branch": branch, "commit": commit,
            "dirty": bool(status) if status is not None else None,
        }
        print(
            "Repository state: "
            f"branch={branch or 'unavailable'} commit={(commit or 'unavailable')[:12]} "
            f"dirty={repository_state['dirty']}"
        )
        print("Baseline stage: configured pre-build checks")
        baseline = [
            _run_command(command, review_config.repo_path, review_config.max_command_output_chars)
            for command in review_config.check_commands
        ]
        baseline_path = run_dir / "baseline.json"
        baseline_path.write_text(json.dumps({
            "repository": repository_state,
            "checks": baseline,
            "passed": all(item["passed"] for item in baseline),
        }, indent=2), encoding="utf-8")
        print(
            f"Baseline: {'passed' if all(item['passed'] for item in baseline) else 'failed'} "
            f"({len(baseline)} checks); {baseline_path.resolve()}"
        )
        print(f"Run log: {log_path}")
        print()
        try:
            result = planning_flow(
                args.request,
                config,
                serena_config,
                context_path=args.context,
                previous_plan_path=args.from_plan,
                run_dir=str(run_dir),
                auto_approve=args.yes,
                answers_path=args.answers,
                context_hints_path=args.context_hints,
                human_input_ledger_path=args.human_input_ledger,
                architecture_decisions_path=args.architecture_decisions,
            )
        except SerenaContextRunError as error:
            return _handle_serena_error(error)
        if result.get("stopped"):
            print()
            print("PLANNING STOPPED: repository context was preserved")
            context_source = result.get("context_source", {})
            context_path = context_source.get("context") or context_source.get("context_path")
            if context_path:
                print(f"Context: {context_path}")
            if result.get("user_input_path"):
                print(f"User input: {result['user_input_path']}")
                print("Fill in the answers and rerun plan with --answers.")
            if result.get("context_input_path"):
                print(f"Context input/artifact: {result['context_input_path']}")
                if str(result["context_input_path"]).endswith("context-input.json"):
                    print("Fill in its hints and rerun plan with --context-hints.")
            if result.get("architecture_input_path"):
                print(f"Architecture input: {result['architecture_input_path']}")
                print("Fill it in and rerun plan with --architecture-decisions.")
            return 0
        plan = result["plan"]
        print()
        print(f"DEVELOPMENT PLAN: {plan['status'].upper()}")
        print(f"Objective: {plan['objective']}")
        print(f"Report: {result['paths']['markdown']}")
        print(f"JSON: {result['paths']['json']}")
        print(f"Log: {result['paths']['log']}")
        if result.get("user_input_path"):
            print(f"User input: {result['user_input_path']}")
            print(
                "After editing it, rerun plan with "
                f"--answers {result['user_input_path']} --from-plan {result['paths']['json']}"
            )
        if args.open_report:
            _open_file(result["paths"]["markdown"])
        if _confirm_open_plan(result["paths"]["json"], args.open_plan):
            _open_file(result["paths"]["json"])
    return 0 if plan["status"] == "ready" else 2


def _run_serena_context(args: argparse.Namespace) -> int:
    config = load_serena_context_config(
        args.repo,
        args.config,
        global_config_path=args.global_config,
        provider_override=args.provider,
        model_override=args.model,
    )
    _print_resolved_config(config)
    try:
        result = serena_context_flow(
            args.request,
            config,
            gate_between_rounds=True,
            auto_approve=False,
        )
    except SerenaContextRunError as error:
        return _handle_serena_error(error)
    report = result["report"]
    print()
    print(f"SERENA CONTEXT: {report['status'].upper()}")
    print(f"Relevant files: {len(report['relevant_files'])}")
    print(f"Context: {result['paths']['context']}")
    print(f"Evidence: {result['paths']['evidence']}")
    if result["paths"].get("preflight"):
        print(f"Preflight: {result['paths']['preflight']}")
    if result["paths"].get("validation_error"):
        print(f"Validation error: {result['paths']['validation_error']}")
    print(f"Round reports: {result['paths']['rounds']}")
    print(f"Transcript: {result['paths']['transcript']}")
    print(f"Log: {result['paths']['log']}")
    return 0 if report["status"] == "sufficient" else 2


def _run_implementation(args: argparse.Namespace) -> int:
    config = load_code_review_config(
        args.repo,
        args.config,
        global_config_path=args.global_config,
        provider_override=args.provider,
        model_override=args.model,
    )
    _print_resolved_config(config)
    result = implementation_flow(args.plan, config, auto_approve=args.yes)
    print(f"Proposal: {result['paths']['proposal']}")
    print(f"Evidence: {result['paths']['evidence']}")
    if result["proposal"]["status"] != "ready":
        return 2
    return 0 if result["applied"] else 2


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "review":
            status = _run_review(args)
        elif args.command == "plan":
            status = _run_plan(args)
        elif args.command == "serena-context":
            status = _run_serena_context(args)
        else:
            status = _run_implementation(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
        return

    sys.exit(status)


if __name__ == "__main__":
    main()
