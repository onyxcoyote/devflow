from langgraph.graph import END, START, StateGraph

from .nodes import prepare_context, review_code, finalize_report
from .state import ReviewState


def build_review_graph():
    builder = StateGraph(ReviewState)

    builder.add_node("prepare_context", prepare_context)
    builder.add_node("review_code", review_code)
    builder.add_node("finalize_report", finalize_report)

    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "review_code")
    builder.add_edge("review_code", "finalize_report")
    builder.add_edge("finalize_report", END)

    return builder.compile()
