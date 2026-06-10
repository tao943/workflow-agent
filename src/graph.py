from functools import partial
from typing import Callable

from langgraph.graph import END, StateGraph

from src.nodes.executor import executor_node
from src.nodes.planner import planner_node
from src.nodes.summarizer import summarizer_node
from src.nodes.verifier import route_after_verifier, verifier_node
from src.state import AgentState
from src.state import MemoryEntry


def _merge_state(state: AgentState, update: dict) -> AgentState:
    merged = dict(state)
    merged.update(update)
    return merged  # type: ignore[return-value]


def _with_checkpoint(node_name: str, node_func: Callable, checkpoint_callback: Callable[[str, AgentState], None] | None):
    def wrapped(state: AgentState) -> dict:
        update = node_func(state)
        if checkpoint_callback is not None:
            checkpoint_callback(node_name, _merge_state(state, update))
        return update

    return wrapped


def build_graph(
    llm=None,
    executor_llm=None,
    verifier_llm=None,
    summarizer_llm=None,
    checkpoint_callback: Callable[[str, AgentState], None] | None = None,
    start_at: str = "planner",
):
    graph = StateGraph(AgentState)

    graph.add_node("planner", _with_checkpoint("planner", partial(planner_node, llm=llm), checkpoint_callback))
    graph.add_node("executor", _with_checkpoint("executor", partial(executor_node, llm=executor_llm), checkpoint_callback))
    graph.add_node("verifier", _with_checkpoint("verifier", partial(verifier_node, llm=verifier_llm), checkpoint_callback))
    graph.add_node("summarizer", _with_checkpoint("summarizer", partial(summarizer_node, llm=summarizer_llm), checkpoint_callback))

    graph.set_entry_point(start_at)
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "executor": "executor",
            "summarizer": "summarizer",
        },
    )
    graph.add_edge("summarizer", END)

    return graph.compile()


def make_initial_state(
    user_task: str,
    memory: list[MemoryEntry] | None = None,
    auto_approve: bool = False,
    session_id: str = "",
    message_id: str = "",
    session_summary: str = "",
    agent: str = "build",
    permission_rules: list[dict[str, str]] | None = None,
    output_format: str = "default",
    storage_path: str = "outputs/agent.sqlite",
    output_dir: str = "outputs",
    context_harness_config: dict | None = None,
    context_namespace: str = "",
    context_snapshot_id: str = "",
    enabled_skills: list[str] | None = None,
    enabled_tools: list[str] | None = None,
    skill_instructions: str = "",
    memory_enabled: bool = True,
) -> AgentState:
    return {
        "user_task": user_task,
        "session_id": session_id,
        "message_id": message_id,
        "session_summary": session_summary,
        "agent": agent,
        "plan": [],
        "current_step_index": 0,
        "agent_step_count": 0,
        "execution_log": [],
        "tool_results": [],
        "memory": memory or [],
        "approvals": [],
        "auto_approve": auto_approve,
        "permission_rules": permission_rules or [],
        "output_format": output_format,
        "storage_path": storage_path,
        "output_dir": output_dir,
        "context_harness_config": context_harness_config or {},
        "context_namespace": context_namespace or session_id,
        "context_snapshot_id": context_snapshot_id,
        "enabled_skills": enabled_skills or [],
        "enabled_tools": enabled_tools or [],
        "skill_instructions": skill_instructions,
        "memory_enabled": memory_enabled,
        "is_complete": False,
        "final_answer": "",
        "error": None,
        "acceptance": {},
    }
