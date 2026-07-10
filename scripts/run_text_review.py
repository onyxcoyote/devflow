# scripts/run_text_review.py

from devflow.text_review.flow import text_review_flow


if __name__ == "__main__":
    text_review_flow(
        "  Hello,    this is my first structured workflow.  "
    )
