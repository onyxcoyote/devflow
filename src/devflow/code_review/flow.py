from prefect import flow

from .config import CodeReviewConfig
from .tasks import collect_change, run_code_review_graph, run_configured_commands, save_review_outputs, validate_repository


@flow(name="code-review")
def code_review_flow(config: CodeReviewConfig) -> dict:
    validate_repository(config)
    change = collect_change(config)
    command_results = run_configured_commands(config)
    final_state = run_code_review_graph({
        "base_ref": config.base_ref, "changed_files": change["changed_files"], "diff": change["diff"],
        "diff_truncated": change["diff_truncated"], "command_results": command_results,
        "review_context": "", "review": {}, "assessment": {}, "report": "",
    }, config)
    paths = save_review_outputs(final_state, config.output_dir)
    return {"assessment": final_state["assessment"], "paths": paths}
