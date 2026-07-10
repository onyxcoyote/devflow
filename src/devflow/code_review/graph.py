from langgraph.graph import END, START, StateGraph

from .nodes import create_report, prepare_review_context, review_code
from .state import CodeReviewState


def build_code_review_graph():
    builder = StateGraph(CodeReviewState)

    builder.add_node("prepare_review_context", prepare_review_context)
    builder.add_node("review_code", review_code)
    builder.add_node("create_report", create_report)

    builder.add_edge(START, "prepare_review_context")
    builder.add_edge("prepare_review_context", "review_code")
    builder.add_edge("review_code", "create_report")
    builder.add_edge("create_report", END)

    return builder.compile()


code_review_graph = build_code_review_graph()
