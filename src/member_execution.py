from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class MemberTaskRequest:
    team_run_id: str
    task_id: str
    logical_task_id: str
    member_name: str
    role: str
    profile: str
    title: str
    description: str
    required_tools: list[str]
    required_evidence: list[str]
    acceptance_criteria: str
    session_id: str
    auto_approve: bool
    output_format: str
    no_memory: bool
    workspace_root: str
    execution_id: str = ""
    idempotency_key: str = ""
    business_attempt: int = 0
    transport_retry_count: int = 0


@dataclass
class MemberExecutionResult:
    member_name: str
    role: str
    status: str
    final_answer: str
    acceptance: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    events: list[dict[str, Any]]
    execution_mode: str
    error: str = ""
    remote_context_id: str = ""
    remote_task_id: str = ""
    trace: dict[str, str] = field(default_factory=dict)


class MemberExecutor(Protocol):
    def execute(self, request: MemberTaskRequest) -> MemberExecutionResult: ...


class LocalMemberExecutor:
    def __init__(self, runtime_factory):
        self.runtime_factory = runtime_factory

    def execute(self, request: MemberTaskRequest) -> MemberExecutionResult:
        runtime = self.runtime_factory(request)
        result = runtime.run(request)
        if isinstance(result, MemberExecutionResult):
            return result
        return MemberExecutionResult(request.member_name, request.role, "completed", str(result), {"passed": True}, [], [], [], "local")
