from langgraph.graph import END, START, StateGraph

from .nodes import create_plan_report, make_plan_node, prepare_plan_context
from .state import PlanningState


def build_planning_graph(model):
    builder = StateGraph(PlanningState)
    builder.add_node("prepare_plan_context", prepare_plan_context)
    builder.add_node("create_plan", make_plan_node(model))
    builder.add_node("create_plan_report", create_plan_report)
    builder.add_edge(START, "prepare_plan_context")
    builder.add_edge("prepare_plan_context", "create_plan")
    builder.add_edge("create_plan", "create_plan_report")
    builder.add_edge("create_plan_report", END)
    return builder.compile()
