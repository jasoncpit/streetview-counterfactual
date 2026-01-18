from pathlib import Path

from langgraph.graph import END, StateGraph

from src.workflow.nodes import criticism, generation, planning, segmentation
from src.workflow.state import AgentState


def build_workflow(cfg, planner, replicate_client):
    """
    Wires the LangGraph workflow based on the provided config and clients.
    """
    graph = StateGraph(AgentState)

    graph.add_node(
        "planner",
        lambda state: planning.plan_edit_node(
            state=state,
            planner=planner,
            target_attribute=cfg.workflow.target_attribute,
        ),
    )
    graph.add_node(
        "segmenter",
        lambda state: segmentation.segment_object_node(
            state=state,
            replicate_client=replicate_client,
            mask_dir=Path(cfg.project.mask_dir),
        ),
    )
    graph.add_node(
        "generator",
        lambda state: generation.inpaint_node(
            state=state,
            replicate_client=replicate_client,
            output_dir=Path(cfg.project.counterfactual_dir),
        ),
    )
    graph.add_node(
        "critic",
        lambda state: criticism.check_realism_node(
            state=state,
            planner=planner,
        ),
    )

    graph.set_entry_point("planner")
    graph.add_edge("planner", "segmenter")
    graph.add_edge("segmenter", "generator")
    graph.add_edge("generator", "critic")

    def realism_router(state: AgentState):
        if state.get("is_realistic"):
            return END
        if state.get("attempts", 0) >= cfg.workflow.max_attempts:
            return END
        return "planner"

    graph.add_conditional_edges("critic", realism_router)
    return graph.compile()

