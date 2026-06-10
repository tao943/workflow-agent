from typing import Any, NotRequired, TypedDict


MAX_EXECUTION_ROUNDS = 5
MAX_AGENT_STEPS = 8
MAX_MEMORY_ITEMS = 50


class PlanStep(TypedDict):
    id: int
    description: str
    status: str
    tool_name: str | None
    tool_args: NotRequired[dict[str, Any] | None]
    result: str | None
    intent: NotRequired[str]
    requires_tool: NotRequired[bool]
    tool_reason: NotRequired[str | None]
    expected_output: NotRequired[str]
    required_evidence: NotRequired[list[str]]
    risk_level: NotRequired[str]
    done_criteria: NotRequired[str]


class MemoryEntry(TypedDict):
    user_task: str
    final_answer: str


class AgentState(TypedDict):
    user_task: str
    session_id: str
    message_id: str
    session_summary: str
    agent: str
    plan: list[PlanStep]
    current_step_index: int
    agent_step_count: int
    execution_log: list[str]
    tool_results: list[dict[str, Any]]
    memory: list[MemoryEntry]
    approvals: list[dict[str, Any]]
    auto_approve: bool
    permission_rules: list[dict[str, str]]
    output_format: str
    storage_path: str
    output_dir: str
    context_harness_config: dict[str, Any]
    context_namespace: str
    context_snapshot_id: str
    feedback_decision: NotRequired[dict[str, Any]]
    enabled_skills: list[str]
    enabled_tools: list[str]
    skill_instructions: str
    memory_enabled: bool
    is_complete: bool
    final_answer: str
    error: str | None
    acceptance: NotRequired[dict[str, Any]]
