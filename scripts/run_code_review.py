import argparse
import shlex

from devflow.code_review.config import CodeReviewConfig
from devflow.code_review.flow import code_review_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a read-only AI review of a Git diff."
    )
    parser.add_argument("--repo", required=True, help="Path to the Git repository")
    parser.add_argument(
        "--base",
        required=True,
        help="Git ref to compare against, for example upstream/beta",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="Trusted check command; repeat for multiple commands",
    )
    parser.add_argument(
        "--test",
        action="append",
        default=[],
        help="Trusted test command; repeat for multiple commands",
    )
    parser.add_argument(
        "--output-dir",
        default="output/code-review",
        help="Directory for review.md, review.json, and evidence.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CodeReviewConfig(
        repo_path=args.repo,
        base_ref=args.base,
        check_commands=tuple(tuple(shlex.split(command)) for command in args.check),
        test_commands=tuple(tuple(shlex.split(command)) for command in args.test),
    )
    code_review_flow(config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
