from langgraph.graph import END, START, StateGraph

from .nodes import assess_review, create_report, make_review_node, prepare_review_context
from .state import CodeReviewState


def build_code_review_graph(model):
    builder = StateGraph(CodeReviewState)
    builder.add_node("prepare_review_context", prepare_review_context)
    builder.add_node("review_code", make_review_node(model))
    builder.add_node("assess_review", assess_review)
    builder.add_node("create_report", create_report)
    builder.add_edge(START, "prepare_review_context")
    builder.add_edge("prepare_review_context", "review_code")
    builder.add_edge("review_code", "assess_review")
    builder.add_edge("assess_review", "create_report")
    builder.add_edge("create_report", END)
    return builder.compile()
