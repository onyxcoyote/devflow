# scripts/draw_text_review.py

from pathlib import Path

from devflow.text_review.graph import text_review_graph


def main() -> None:
    output = Path("diagrams/text-review.mmd")
    output.parent.mkdir(parents=True, exist_ok=True)

    mermaid = text_review_graph.get_graph().draw_mermaid()
    output.write_text(mermaid, encoding="utf-8")

    print(f"Wrote diagram to: {output}")


if __name__ == "__main__":
    main()
