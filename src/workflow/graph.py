from pathlib import Path

from langgraph.graph import END, StateGraph

from src.workflow.nodes import criticism, generation, planning, segmentation
from src.workflow.state import AgentState

def build_baseline_workflow(cfg, planner, replicate_client):
    graph = StateGraph(AgentState)
    graph.add_node(
        "planner",
        lambda state: planning.plan_edit_node(
            state=state,
            planner=planner,
            target_attribute=cfg.workflow.target_attribute,
        ),
    )
    def baseline_node(state: AgentState):
        edited_path = replicate_client.image_edit_baseline(
            output_dir=Path(cfg.project.baseline_dir),
            model=cfg.workflow.baseline_model,
            image_path=state["image_path"],
            edit_plan=state["edit_plan"],
            target_object=state["target_object"],
        )
        if not edited_path:
            raise ValueError("Baseline edit failed to produce an output image.")
        return {
            "edited_image_path": str(edited_path),
            "used_mock": replicate_client.last_baseline_used_mock,
        }

    graph.add_node("baseline", baseline_node)
    graph.add_node(
        "critic",
        lambda state: criticism.critique_generated_node(
            state=state,
            planner=planner,
        ),
    )
    graph.add_edge("planner", "baseline")
    graph.add_edge("baseline", "critic")
    def critic_router(state: AgentState):
        if state.get("is_realistic") and state.get("is_minimal_edit"):
            return END
        if state.get("attempts", 0) >= cfg.workflow.max_attempts:
            return END
        return "planner"

    graph.add_conditional_edges("critic", critic_router)
    graph.set_entry_point("planner")
    return graph.compile()

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
        lambda state: criticism.critique_generated_node(
            state=state,
            planner=planner,
        ),
    )

    graph.set_entry_point("planner")
    graph.add_edge("planner", "segmenter")
    graph.add_edge("segmenter", "generator")
    graph.add_edge("generator", "critic")
    def critic_router(state: AgentState):
        if state.get("is_realistic") and state.get("is_minimal_edit"):
            return END
        if state.get("attempts", 0) >= cfg.workflow.max_attempts:
            return END
        return "planner"

    graph.add_conditional_edges("critic", critic_router)

    return graph.compile()
