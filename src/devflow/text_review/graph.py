# src/devflow/text_review/graph.py

from langgraph.graph import END, START, StateGraph

from .nodes import (
    analyze_text,
    create_report,
    normalize_text,
    review_text_with_ai,
    route_after_review,
    suggest_improvement,
)
from .state import TextReviewState


def build_text_review_graph():
    builder = StateGraph(TextReviewState)

    builder.add_node("normalize_text", normalize_text)
    builder.add_node("analyze_text", analyze_text)
    builder.add_node("review_text_with_ai", review_text_with_ai)
    builder.add_node("suggest_improvement", suggest_improvement)
    builder.add_node("create_report", create_report)

    builder.add_edge(START, "normalize_text")
    builder.add_edge("normalize_text", "analyze_text")
    builder.add_edge("analyze_text", "review_text_with_ai")

    builder.add_conditional_edges(
        "review_text_with_ai",
        route_after_review,
        {
            "suggest_improvement": "suggest_improvement",
            "create_report": "create_report",
        },
    )

    builder.add_edge("suggest_improvement", "create_report")
    builder.add_edge("create_report", END)

    return builder.compile()


text_review_graph = build_text_review_graph()
