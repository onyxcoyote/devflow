# src/devflow/text_review/flow.py

from prefect import flow

from .tasks import prepare_input, run_review_graph, save_report


@flow(name="text-review")
def text_review_flow(
    text: str,
    output_path: str = "output/text-review.txt",
) -> str:
    initial_state = prepare_input(text)
    final_state = run_review_graph(initial_state)

    saved_path = save_report(
        final_state["report"],
        output_path,
    )

    print(final_state["report"])
    print(f"Saved report to: {saved_path}")

    return saved_path
