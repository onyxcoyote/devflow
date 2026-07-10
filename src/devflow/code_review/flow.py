from prefect import flow

from .config import CodeReviewConfig
from .tasks import (
    collect_change,
    run_code_review_graph,
    run_configured_commands,
    save_review_outputs,
    validate_repository,
)


@flow(name="code-review")
def code_review_flow(
    config: CodeReviewConfig,
    output_dir: str = "output/code-review",
) -> dict[str, str]:
    validate_repository(config)
    change = collect_change(config)
    command_results = run_configured_commands(config)

    initial_state = {
        "base_ref": config.base_ref,
        "changed_files": change["changed_files"],
        "diff": change["diff"],
        "diff_truncated": change["diff_truncated"],
        "command_results": command_results,
        "review_context": "",
        "review": {},
        "report": "",
    }

    final_state = run_code_review_graph(initial_state)
    paths = save_review_outputs(final_state, output_dir)

    print(final_state["report"])
    print(f"Saved Markdown review to: {paths['markdown']}")
    print(f"Saved JSON review to: {paths['json']}")
    print(f"Saved evidence to: {paths['evidence']}")

    return paths
