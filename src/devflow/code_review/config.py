from dataclasses import dataclass


@dataclass(frozen=True)
class CodeReviewConfig:
    """Limits and repository settings for a code-review run."""

    repo_path: str
    base_ref: str
    check_commands: tuple[tuple[str, ...], ...] = ()
    test_commands: tuple[tuple[str, ...], ...] = ()
    max_diff_chars: int = 40_000
    max_command_output_chars: int = 12_000
