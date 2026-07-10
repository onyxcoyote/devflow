from pathlib import Path

from devflow.code_review.graph import code_review_graph


def main() -> None:
    output = Path("diagrams/code-review.mmd")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        code_review_graph.get_graph().draw_mermaid(),
        encoding="utf-8",
    )
    print(f"Wrote diagram to: {output}")


if __name__ == "__main__":
    main()
