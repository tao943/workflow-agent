from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.assignment import AssignmentPlanner, GraphReplanner, TaskGraph, TaskNode, build_member_capabilities
from src.config import AppConfig, AgentSpec, PermissionRule, TeamSpec, get_team_spec
from src.context_harness import ContextHarness, MemberReport
from src.evidence import verify_task_evidence
from src.memory import load_memory, remember_interaction, save_memory
from src.memory_runtime import MemoryGovernanceRuntime
from src.permissions import evaluate_permission, request_permission
from src.project_files import ProjectFileFilter
from src.prompts.team_prompt import TEAM_LEAD_SYSTEM_PROMPT
from src.prompts.templates import PROMPT_VERSION
from src.runtime import RuntimeOptions
from src.session import create_or_continue_session, new_id, new_session_id
from src.storage import Storage
from src.synthesis import LeadSynthesizer
from src.tool_runtime import ToolRunRequest, ToolRuntime
from src.tools.registry import ToolContext


@dataclass
class TeamRunResult:
    team_run_id: str
    team_id: str
    session_id: str
    status: str
    final_answer: str
    events: list[dict[str, Any]] = field(default_factory=list)
    members: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    member_runs: list[dict[str, Any]] = field(default_factory=list)
    learnings: dict[str, Any] = field(default_factory=dict)


class TeamRuntime:
    def __init__(self, config: AppConfig, llm=None, storage: Storage | None = None, agent_runtime_factory=None) -> None:
        self.config = config
        self.llm = llm
        self.storage = storage or Storage(self.storage_path)
        self.agent_runtime_factory = agent_runtime_factory

    @property
    def storage_path(self) -> str:
        return str(self.config.output_dir.rstrip("/\\") + "/agent.sqlite")

    @property
    def memory_path(self) -> str:
        return str(self.config.output_dir.rstrip("/\\") + "/memory.sqlite")

    def run(self, task: str, team_name: str, auto_approve: bool = False, output_format: str = "default", no_memory: bool = False) -> TeamRunResult:
        spec = get_team_spec(team_name)
        self._validate_team(spec)

        lead_session = create_or_continue_session(
            storage=self.storage,
            task=task,
            agent=spec.lead.name,
            output_format=output_format,
            session_id=None,
            parent_id=None,
        )
        team_id = f"team_{spec.name}"
        team_run_id = new_id("team_run")
        self._require_permission(lead_session.session_id, "team:create", spec.name, auto_approve, output_format)
        self.storage.create_team(team_id, spec.name, spec.description, spec.model_dump())
        self.storage.create_team_run(team_run_id, team_id, lead_session.session_id, task)
        self.storage.add_event(lead_session.session_id, "team.created", {"team_run_id": team_run_id, "team": spec.name})
        if self.config.memory.enabled and not no_memory:
            MemoryGovernanceRuntime(
                self.memory_path,
                storage=self.storage,
                min_active_confidence=self.config.memory.min_active_confidence,
            ).hot_write_user_task(task, namespace=self.config.memory.namespace, session_id=lead_session.session_id)

        self._create_members(team_run_id, lead_session.session_id, spec)
        assignment_graph = AssignmentPlanner(self.config, self.llm).plan(task, spec)
        assignment = assignment_graph.model_dump()
        self.storage.add_team_message(
            new_id("team_msg"),
            team_run_id,
            spec.lead.name,
            "team",
            {"type": "assignment.selected", **assignment},
        )
        self.storage.add_event(lead_session.session_id, "team.assignment.selected", assignment)
        self.storage.add_event(lead_session.session_id, "team.route.selected", assignment)
        self.storage.add_event(lead_session.session_id, "team.graph.validated", {"team_run_id": team_run_id, "strategy": assignment_graph.strategy, "issues": assignment_graph.issues})
        if assignment_graph.fallback_used:
            self.storage.add_event(lead_session.session_id, "team.graph.fallback", {"team_run_id": team_run_id, "issues": assignment_graph.issues})
        member_tasks = self._create_tasks(team_run_id, task, spec, assignment_graph)
        member_results = self._run_members(team_run_id, lead_session.session_id, member_tasks, spec, task, auto_approve, output_format, no_memory)
        final_answer, learnings = self._summarize(task, spec, member_results)
        self.storage.add_team_message(
            new_id("team_msg"),
            team_run_id,
            spec.lead.name,
            "user",
            {"type": "final_answer", "content": final_answer, "learnings": learnings},
        )
        self.storage.add_event(lead_session.session_id, "team.completed", {"team_run_id": team_run_id, "learnings": learnings})
        self.storage.update_team_run(team_run_id, "completed", final_answer)
        self.storage.update_session_status(lead_session.session_id, "completed")

        if self.config.memory.enabled and not no_memory:
            memory = load_memory(self.memory_path)
            next_memory = remember_interaction(memory, f"[team:{spec.name}] {task}", final_answer, [])
            save_memory(next_memory, self.memory_path)
            MemoryGovernanceRuntime(
                self.memory_path,
                storage=self.storage,
                min_active_confidence=self.config.memory.min_active_confidence,
            ).consolidate_run(
                user_task=f"[team:{spec.name}] {task}",
                final_answer=final_answer,
                tool_results=self._team_evidence_summary(team_run_id),
                acceptance={"passed": True, "issues": []},
                namespace=self.config.memory.namespace,
                session_id=lead_session.session_id,
                learnings=learnings,
            )

        return self._result(team_run_id, team_id, lead_session.session_id, final_answer, learnings)

    def inspect(self, team_run_id: str) -> TeamRunResult:
        run = self.storage.get_team_run(team_run_id)
        if not run:
            raise ValueError(f"Unknown team run: {team_run_id}")
        return self._result(
            team_run_id,
            run["team_id"],
            run["session_id"],
            run.get("final_answer") or "",
            {},
            status=run["status"],
        )

    def _validate_team(self, spec: TeamSpec) -> None:
        if len(spec.members) > self.config.team_mode.max_members:
            raise ValueError(f"Team {spec.name} has too many members.")

    def _require_permission(self, session_id: str, permission: str, pattern: str, auto_approve: bool, output_format: str) -> None:
        rules = [
            PermissionRule(permission="team:create", pattern="*", action="ask"),
            PermissionRule(permission="team:message", pattern="*", action="allow"),
            PermissionRule(permission="team:task", pattern="*", action="allow"),
            PermissionRule(permission="team:delete", pattern="*", action="ask"),
            PermissionRule(permission="team:external_worktree", pattern="*", action="deny"),
            *self.config.permissions,
        ]
        permission_name = permission
        decision = evaluate_permission(permission_name, pattern or "*", rules)
        if not request_permission(session_id, self.storage, decision, f"Team action {permission}", auto_approve, output_format):
            raise PermissionError(f"Permission denied for {permission}:{pattern}")

    def _create_members(self, team_run_id: str, parent_session_id: str, spec: TeamSpec) -> None:
        for member in [spec.lead, *spec.members]:
            session_id = new_session_id()
            self.storage.create_session(session_id, f"{spec.name}:{member.name}", member.profile, parent_id=parent_session_id)
            self.storage.add_team_member(new_id("team_member"), team_run_id, member.name, member.role, member.profile, session_id)
            self.storage.add_event(parent_session_id, "team.member.created", {"team_run_id": team_run_id, "member": member.name, "role": member.role})

    def _create_tasks(self, team_run_id: str, task: str, spec: TeamSpec, assignment_graph: TaskGraph) -> list[dict[str, Any]]:
        tasks = []
        members_by_name = {member.name: member for member in spec.members}
        for node in assignment_graph.tasks:
            member = members_by_name[node.assigned_to]
            task_id = new_id("team_task")
            self.storage.add_team_task(
                task_id,
                team_run_id,
                node.title,
                node.description,
                member.name,
                required_tools=node.required_tools,
                required_evidence=node.required_evidence,
                evidence_status="missing" if node.required_tools else "passed",
                metadata={
                    "logical_id": node.id,
                    "depends_on": node.depends_on,
                    "assignment_reason": node.assignment_reason,
                    "assignment_strategy": assignment_graph.strategy,
                    "acceptance_criteria": node.acceptance_criteria,
                    "risk_level": node.risk_level,
                    "intent": assignment_graph.intent,
                    "node_type": node.node_type,
                    "optional": node.optional,
                    "condition": node.condition,
                    "max_retries": node.max_retries,
                    "replan_policy": node.replan_policy,
                    "why_this_member": node.why_this_member,
                    "why_now": node.why_now,
                    "skip_condition": node.skip_condition,
                    "replan_round": node.replan_round,
                    **(node.metadata or {}),
                },
            )
            self.storage.add_team_message(
                new_id("team_msg"),
                team_run_id,
                spec.lead.name,
                member.name,
                {
                    "type": "task",
                    "task_id": task_id,
                    "logical_id": node.id,
                    "title": node.title,
                    "description": node.description,
                    "depends_on": node.depends_on,
                    "required_tools": node.required_tools,
                    "required_evidence": node.required_evidence,
                    "assignment_strategy": assignment_graph.strategy,
                    "assignment_reason": node.assignment_reason,
                    "node_type": node.node_type,
                    "condition": node.condition,
                    "optional": node.optional,
                    "metadata": node.metadata or {},
                },
            )
            self.storage.add_event(
                team_run_id,
                "team.task.created",
                {"task_id": task_id, "logical_id": node.id, "assigned_to": member.name, "title": node.title, "depends_on": node.depends_on, "required_tools": node.required_tools, "assignment_strategy": assignment_graph.strategy},
            )
            tasks.append(
                {
                    "id": task_id,
                    "logical_id": node.id,
                    "member": member,
                    "title": node.title,
                    "description": node.description,
                    "depends_on": node.depends_on,
                    "required_tools": node.required_tools,
                    "required_evidence": node.required_evidence,
                    "acceptance_criteria": node.acceptance_criteria,
                    "assignment_strategy": assignment_graph.strategy,
                    "node_type": node.node_type,
                    "condition": node.condition,
                    "optional": node.optional,
                    "replan_round": node.replan_round,
                    "metadata": node.metadata or {},
                }
            )
        return tasks

    def _create_replanned_tasks(self, team_run_id: str, spec: TeamSpec, nodes: list[TaskNode], strategy: str) -> list[dict[str, Any]]:
        graph = TaskGraph(intent="mixed", strategy=strategy, tasks=nodes)
        return self._create_tasks(team_run_id, "", spec, graph)

    def _task_graph_from_runtime_items(self, items: list[dict[str, Any]]) -> TaskGraph:
        return TaskGraph(intent="mixed", strategy=items[0].get("assignment_strategy", "baseline") if items else "baseline", tasks=[self._task_node_from_runtime_item(item) for item in items])

    def _task_node_from_runtime_item(self, item: dict[str, Any]) -> TaskNode:
        member: AgentSpec = item["member"]
        return TaskNode(
            id=item["logical_id"],
            title=item["title"],
            description=item["description"],
            assigned_to=member.name,
            depends_on=list(item.get("depends_on") or []),
            required_tools=list(item.get("required_tools") or []),
            required_evidence=list(item.get("required_evidence") or []),
            acceptance_criteria=str(item.get("acceptance_criteria") or ""),
            node_type=item.get("node_type", "task"),
            optional=bool(item.get("optional", False)),
            condition=item.get("condition", "always"),
            replan_round=int(item.get("replan_round", 0) or 0),
            metadata=dict(item.get("metadata") or {}),
        )

    def _task_title(self, member: AgentSpec, task: str) -> str:
        if member.role == "researcher":
            return "Research context and constraints"
        if member.role == "planner":
            return "Create implementation plan"
        if member.role == "builder":
            return "Execute build-oriented work"
        if member.role == "reviewer":
            return "Review outputs and acceptance"
        return f"Handle {member.role} work"

    def _required_tools(self, member: AgentSpec, task: str) -> list[str]:
        if member.role == "researcher" and any(keyword in task.lower() for keyword in ("项目", "project", "代码", "结构", "repo")):
            return ["list_files", "read_file", "search_files"]
        return []

    def _required_evidence(self, member: AgentSpec, task: str) -> list[str]:
        if member.role == "researcher" and self._required_tools(member, task):
            return ["project file list", "key source files", "code search matches"]
        if member.role in {"planner", "reviewer"}:
            return ["researcher evidence or tool call summary"]
        return []

    def _task_description(self, member: AgentSpec, task: str, required_tools: list[str], required_evidence: list[str]) -> str:
        return "\n".join(
            [
                f"Prompt Version: {PROMPT_VERSION}",
                member.system_prompt or f"Act as {member.role}.",
                f"User task: {task}",
                f"Required tools: {required_tools or 'none'}",
                f"Required evidence: {required_evidence or 'none'}",
                "Do not claim that you scanned, read, searched, saved, or wrote anything unless the evidence summary includes the matching tool result.",
                "Return concise findings, completed work, risks, and next steps.",
            ]
        )

    def _run_members(
        self,
        team_run_id: str,
        lead_session_id: str,
        tasks: list[dict[str, Any]],
        spec: TeamSpec,
        user_task: str,
        auto_approve: bool,
        output_format: str,
        no_memory: bool,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        pending = {item["logical_id"]: item for item in tasks}
        completed: set[str] = set()
        failed: set[str] = set()
        graph = self._task_graph_from_runtime_items(tasks)
        replanner = GraphReplanner(self.config, build_member_capabilities(spec))
        replan_round = 0
        while pending:
            blocked = [
                item
                for item in pending.values()
                if set(item.get("depends_on") or []).intersection(failed)
            ]
            for item in blocked:
                payload = self._blocked_member_result(team_run_id, item)
                results.append(payload)
                failed.add(item["logical_id"])
                pending.pop(item["logical_id"], None)

            ready = [
                item
                for item in pending.values()
                if set(item.get("depends_on") or []).issubset(completed)
            ]
            skipped_ready = []
            for item in ready:
                skip_reason = replanner.should_skip(self._task_node_from_runtime_item(item), user_task, completed, failed)
                if skip_reason and item.get("optional"):
                    payload = self._blocked_member_result(team_run_id, item, reason=skip_reason, status="skipped")
                    results.append(payload)
                    completed.add(item["logical_id"])
                    pending.pop(item["logical_id"], None)
                    skipped_ready.append(item)
                    self.storage.add_event(lead_session_id, "team.task.skipped", {"team_run_id": team_run_id, "task_id": item["id"], "logical_id": item["logical_id"], "reason": skip_reason})
            if skipped_ready:
                ready = [item for item in ready if item not in skipped_ready]
            if not ready:
                if pending and skipped_ready:
                    continue
                for item in list(pending.values()):
                    payload = self._blocked_member_result(team_run_id, item, reason="dependency cycle or missing dependency")
                    results.append(payload)
                    failed.add(item["logical_id"])
                    pending.pop(item["logical_id"], None)
                break

            max_workers = max(1, min(self.config.team_mode.max_parallel_members, len(ready)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._run_member_task, team_run_id, lead_session_id, item, auto_approve, output_format, no_memory): item
                    for item in ready
                }
                for future in as_completed(futures):
                    item = futures[future]
                    payload = future.result()
                    results.append(payload)
                    if payload.get("status") == "completed" and payload.get("evidence_check", {}).get("passed", True):
                        completed.add(item["logical_id"])
                    else:
                        failed.add(item["logical_id"])
                    pending.pop(item["logical_id"], None)

            if self.config.team_mode.replan_strategy in {"after_layer", "after_failure"}:
                if self.config.team_mode.replan_strategy == "after_failure" and not failed:
                    continue
                additions = replanner.maybe_replan(graph, completed, failed, results, replan_round)
                if additions:
                    replan_round += 1
                    graph.tasks.extend(additions)
                    new_items = self._create_replanned_tasks(team_run_id, spec, additions, graph.strategy)
                    for item in new_items:
                        pending[item["logical_id"]] = item
                    self.storage.add_event(
                        lead_session_id,
                        "team.graph.replanned",
                        {"team_run_id": team_run_id, "round": replan_round, "tasks": [task.__dict__ for task in additions]},
                    )
        results.sort(key=lambda row: row["member"])
        return results

    def _blocked_member_result(self, team_run_id: str, item: dict[str, Any], reason: str = "upstream dependency failed", status: str = "skipped") -> dict[str, Any]:
        member: AgentSpec = item["member"]
        session_id = self._member_session_id(team_run_id, member.name)
        final_answer = f"Task skipped: {reason}."
        payload = {
            "member": member.name,
            "role": member.role,
            "status": status,
            "session_id": session_id,
            "final_answer": final_answer,
            "acceptance": {"passed": False, "issues": [reason]},
            "artifacts": [],
            "evidence": [],
            "evidence_check": {"passed": False, "status": "missing", "issues": [reason]},
        }
        self.storage.update_team_task(item["id"], status, final_answer, member.name, "missing")
        self.storage.update_team_task_metadata(item["id"], {"skip_reason": reason})
        self.storage.update_team_member(team_run_id, member.name, status)
        self.storage.add_team_message(new_id("team_msg"), team_run_id, member.name, "coordinator", {"type": "result", **payload})
        self.storage.add_team_member_run(new_id("member_run"), team_run_id, member.name, session_id, status, payload)
        return payload

    def _run_member_task(
        self,
        team_run_id: str,
        lead_session_id: str,
        item: dict[str, Any],
        auto_approve: bool,
        output_format: str,
        no_memory: bool,
    ) -> dict[str, Any]:
        member: AgentSpec = item["member"]
        self.storage.update_team_task(item["id"], "running", assigned_to=member.name)
        self.storage.update_team_member(team_run_id, member.name, "running")
        self.storage.add_event(lead_session_id, "team.task.claimed", {"team_run_id": team_run_id, "task_id": item["id"], "member": member.name})

        evidence = self._collect_required_evidence(
            team_run_id=team_run_id,
            task_id=item["id"],
            member=member,
            required_tools=item.get("required_tools") or [],
            query=item.get("description") or item.get("title") or "",
            evidence_plan=(item.get("metadata") or {}).get("evidence_plan"),
            auto_approve=auto_approve,
            output_format=output_format,
        )
        if item.get("required_tools") and member.role == "researcher":
            session_id = self._member_session_id(team_run_id, member.name)
            payload = {
                "member": member.name,
                "role": member.role,
                "status": "completed",
                "session_id": session_id,
                "final_answer": self._evidence_member_answer(item, evidence),
                "acceptance": {"passed": True, "issues": []},
                "artifacts": [item["raw_output_path"] for item in evidence if item.get("raw_output_path")],
                "evidence": evidence,
                "task_metadata": item.get("metadata") or {},
            }
            tool_calls = self.storage.list_tool_calls(session_id)
            evidence_check = verify_task_evidence(item, payload, tool_calls)
            payload["evidence_check"] = {
                "passed": evidence_check.passed,
                "status": evidence_check.status,
                "issues": evidence_check.issues,
            }
            task_status = "completed" if evidence_check.passed else "error"
            self.storage.update_team_task(item["id"], task_status, payload["final_answer"], member.name, evidence_check.status)
            self.storage.update_team_member(team_run_id, member.name, task_status)
            self.storage.add_team_message(new_id("team_msg"), team_run_id, member.name, "coordinator", {"type": "result", **payload})
            self.storage.add_team_member_run(new_id("member_run"), team_run_id, member.name, session_id, task_status, payload)
            self.storage.add_event(lead_session_id, "team.member.completed", {"team_run_id": team_run_id, "member": member.name, "status": task_status, "evidence": payload["evidence_check"]})
            self.storage.add_event(lead_session_id, "team.task.completed", {"team_run_id": team_run_id, "task_id": item["id"], "member": member.name})
            return payload

        if item.get("assignment_strategy") == "claw_baseline":
            session_id = self._member_session_id(team_run_id, member.name)
            inherited_evidence = evidence or self._team_evidence_summary(team_run_id)
            final_answer = self._evidence_member_answer(item, inherited_evidence) if inherited_evidence else f"Claw task {item['title']} completed from prior evidence."
            payload = {
                "member": member.name,
                "role": member.role,
                "status": "completed",
                "session_id": session_id,
                "final_answer": final_answer,
                "acceptance": {"passed": True, "issues": []},
                "artifacts": [evidence_item["raw_output_path"] for evidence_item in inherited_evidence if evidence_item.get("raw_output_path")],
                "evidence": inherited_evidence,
                "task_metadata": item.get("metadata") or {},
            }
            payload["member_report"] = self._member_report(item, payload)
            evidence_check = verify_task_evidence(item, payload, self.storage.list_tool_calls(session_id))
            payload["evidence_check"] = {
                "passed": evidence_check.passed,
                "status": evidence_check.status,
                "issues": evidence_check.issues,
            }
            task_status = "completed" if evidence_check.passed else "error"
            self.storage.update_team_task(item["id"], task_status, final_answer, member.name, evidence_check.status)
            self.storage.update_team_member(team_run_id, member.name, task_status)
            self.storage.add_team_message(new_id("team_msg"), team_run_id, member.name, "coordinator", {"type": "result", **payload})
            self.storage.add_team_member_run(new_id("member_run"), team_run_id, member.name, session_id, task_status, payload)
            self.storage.add_event(lead_session_id, "team.member.completed", {"team_run_id": team_run_id, "member": member.name, "status": task_status, "evidence": payload["evidence_check"]})
            self.storage.add_event(lead_session_id, "team.task.completed", {"team_run_id": team_run_id, "task_id": item["id"], "member": member.name})
            return payload

        member_config = self.config.model_copy(update={"default_agent": member.profile})
        remote_cfg = self.config.a2a.agents.get(member.role) if hasattr(self.config, "a2a") else None
        if self.config.a2a.enabled and remote_cfg and remote_cfg.enabled and member.role in {"researcher", "builder", "reviewer"}:
            from src.a2a_runtime import A2AAgentRegistry
            import httpx
            session_id = self._member_session_id(team_run_id, member.name)
            try:
                remote = A2AAgentRegistry(self.config.a2a).resolve(member.role)
                response = httpx.post(remote.base_origin + "/a2a/rest/v1/message:send", headers={"Authorization": remote.authorization_header}, json={"role": member.role, "task": item["description"], "team_run_id": team_run_id, "task_id": item["logical_id"]}, timeout=self.config.a2a.request_timeout_seconds)
                response.raise_for_status()
                body = response.json()
                payload = {"member": member.name, "role": member.role, "status": body.get("status", "completed"), "session_id": session_id, "final_answer": body.get("result", ""), "acceptance": {"passed": True, "issues": []}, "artifacts": [], "evidence": evidence, "execution_mode": "a2a", "task_metadata": item.get("metadata") or {}}
                payload["member_report"] = self._member_report(item, payload)
                self.storage.update_team_task(item["id"], "completed", str(payload["final_answer"]), member.name, "satisfied")
                self.storage.update_team_member(team_run_id, member.name, "completed")
                self.storage.add_team_member_run(new_id("member_run"), team_run_id, member.name, session_id, "completed", payload)
                return payload
            except Exception as exc:
                if not (member.role in {"researcher", "reviewer"} and remote_cfg.allow_local_fallback):
                    self.storage.update_team_task(item["id"], "blocked", "", member.name, "missing")
                    self.storage.update_team_member(team_run_id, member.name, "blocked")
                    return {"member": member.name, "role": member.role, "status": "blocked", "session_id": session_id, "final_answer": "", "acceptance": {"passed": False, "issues": [str(exc)]}, "artifacts": [], "evidence": evidence, "execution_mode": "a2a", "error": str(exc)}
        runtime = self._agent_runtime(member_config)
        inherited_evidence = evidence or self._team_evidence_summary(team_run_id)
        task_description = self._with_evidence_summary(item["description"], inherited_evidence)
        result = runtime.run(
            RuntimeOptions(
                task=task_description,
                session_id=self._member_session_id(team_run_id, member.name),
                auto_approve=auto_approve,
                output_format=output_format,
                no_memory=no_memory,
                full_model=False,
                context_namespace=f"context://{team_run_id}/{member.name}/{item['logical_id']}",
                consolidate_memory=False,
            )
        )
        payload = {
            "member": member.name,
            "role": member.role,
            "status": result.status,
            "session_id": result.session_id,
            "final_answer": result.final_answer,
            "acceptance": result.acceptance,
            "artifacts": result.artifacts,
            "evidence": evidence,
            "task_metadata": item.get("metadata") or {},
        }
        payload["member_report"] = self._member_report(item, payload)
        tool_calls = self.storage.list_tool_calls(result.session_id)
        evidence_check = verify_task_evidence(item, payload, tool_calls)
        payload["evidence_check"] = {
            "passed": evidence_check.passed,
            "status": evidence_check.status,
            "issues": evidence_check.issues,
        }
        task_status = "completed" if result.status == "completed" and evidence_check.passed else "error"
        self.storage.update_team_task(item["id"], task_status, result.final_answer, member.name, evidence_check.status)
        self.storage.update_team_member(team_run_id, member.name, task_status)
        self.storage.add_team_message(new_id("team_msg"), team_run_id, member.name, "coordinator", {"type": "result", **payload})
        self.storage.add_team_member_run(new_id("member_run"), team_run_id, member.name, result.session_id, task_status, payload)
        self.storage.add_event(lead_session_id, "team.member.completed", {"team_run_id": team_run_id, "member": member.name, "status": task_status, "evidence": payload["evidence_check"]})
        self.storage.add_event(lead_session_id, "team.task.completed", {"team_run_id": team_run_id, "task_id": item["id"], "member": member.name})
        return payload

    def _member_report(self, item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        evidence_ids = [
            f"ev_{index}"
            for index, evidence in enumerate(payload.get("evidence") or [], start=1)
            if evidence.get("status") == "success"
        ]
        claims = [
            {"claim": line.strip("- "), "evidence_ids": evidence_ids}
            for line in str(payload.get("final_answer") or "").splitlines()
            if line.strip().startswith("-") and evidence_ids
        ][:8]
        return MemberReport(
            task_id=item["logical_id"],
            status=payload.get("status", "unknown"),
            claims=claims,
            evidence_ids=evidence_ids,
            artifacts=list(payload.get("artifacts") or []),
            failed_attempts=list((payload.get("acceptance") or {}).get("issues") or []),
            open_issues=list((payload.get("evidence_check") or {}).get("issues") or []),
        ).__dict__

    def _collect_required_evidence(
        self,
        team_run_id: str,
        task_id: str,
        member: AgentSpec,
        required_tools: list[str],
        query: str,
        evidence_plan: dict[str, Any] | None,
        auto_approve: bool,
        output_format: str,
    ) -> list[dict[str, Any]]:
        if not required_tools:
            return []
        session_id = self._member_session_id(team_run_id, member.name)
        evidence: list[dict[str, Any]] = []
        for tool_name, args in self._evidence_tool_requests(required_tools, query, evidence_plan):
            result, _ = ToolRuntime(artifact_dir=Path(self.config.output_dir) / "artifacts").run(
                ToolRunRequest(
                    name=tool_name,
                    args=args,
                    context=ToolContext(session_id=session_id, step_id=task_id, output_dir=Path(self.config.output_dir)),
                    session_id=session_id,
                    step_id=task_id,
                    enabled_tools=set(required_tools),
                    permission_rules=self.config.permissions,
                    auto_approve=auto_approve,
                    output_format=output_format,
                    storage=self.storage,
                )
            )
            evidence_item = {
                "tool": tool_name,
                "args": args,
                "status": result.status,
                "output": (result.display_output or result.output or "")[:1500],
                "error": result.error,
                "metadata": result.metadata,
                "raw_output_path": result.raw_output_path,
            }
            evidence.append(evidence_item)
            self.storage.add_team_message(new_id("team_msg"), team_run_id, member.name, "team", {"type": "evidence", "task_id": task_id, **evidence_item})
        return evidence

    def _evidence_tool_requests(self, required_tools: list[str], query: str = "", evidence_plan: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
        requests: list[tuple[str, dict[str, Any]]] = []
        if "list_files" in required_tools:
            requests.append(("list_files", {"path": "."}))
        if "read_file" in required_tools:
            for path in self._planned_key_files(evidence_plan):
                requests.append(("read_file", {"path": path}))
        if "search_files" in required_tools:
            for search_query in self._planned_search_queries(evidence_plan):
                requests.append(("search_files", {"query": search_query, "path": "src"}))
        if "rag_search" in required_tools:
            requests.append(("rag_search", {"query": query or "knowledge base context", "top_k": 5}))
        if "kb_search" in required_tools:
            requests.append(("kb_search", {"query": "VPN remote work device security password", "max_results": 10}))
        if "kb_get_article" in required_tools:
            for article_id in self._claw_article_ids(query):
                requests.append(("kb_get_article", {"article_id": article_id}))
        if "notes_list" in required_tools:
            requests.append(("notes_list", {"max_results": 10}))
        if "notes_get" in required_tools:
            for note_id in self._claw_note_ids(query):
                requests.append(("notes_get", {"note_id": note_id}))
        if "notes_share" in required_tools:
            requests.append(
                (
                    "notes_share",
                    {
                        "note_id": "note_001",
                        "recipients": ["Manager Zhang", "Li Ming", "Wang Fang", "Zhao Qiang"],
                    },
                )
            )
        if "contacts_search" in required_tools:
            requests.append(("contacts_search", {"query": "David Zhang", "department": "Engineering"}))
        if "contacts_get" in required_tools:
            for contact_id in self._claw_contact_ids(query):
                requests.append(("contacts_get", {"contact_id": contact_id}))
        if "write_file" in required_tools:
            requests.append(
                (
                    "write_file",
                    {
                        "path": self._extract_output_path(query) or "outputs/team_build_output.md",
                        "content": self._extract_write_content(query),
                    },
                )
            )
        return requests

    def _claw_article_ids(self, query: str) -> list[str]:
        ids = self._unique_matches(r"\bkb_\d{3}\b", query)
        return ids or ["kb_001", "kb_002", "kb_003", "kb_005", "kb_006", "kb_007"]

    def _claw_note_ids(self, query: str) -> list[str]:
        ids = self._unique_matches(r"\bnote_\d{3}\b", query)
        return ids or ["note_001", "note_002", "note_004"]

    def _claw_contact_ids(self, query: str) -> list[str]:
        ids = self._unique_matches(r"\bc_\d{3}\b", query)
        return ids or ["c_001", "c_007", "c_002", "c_003"]

    def _unique_matches(self, pattern: str, text: str) -> list[str]:
        values: list[str] = []
        for value in re.findall(pattern, text):
            if value not in values:
                values.append(value)
        return values

    def _planned_key_files(self, evidence_plan: dict[str, Any] | None) -> list[str]:
        planned = list((evidence_plan or {}).get("suggested_files") or [])
        if not planned:
            return self._existing_key_files()
        project_filter = ProjectFileFilter()
        selected: list[str] = []
        for value in planned:
            path = Path(str(value))
            resolved = path if path.is_absolute() else project_filter.workspace / path
            if resolved.exists() and project_filter.should_include(resolved):
                selected.append(str(value).replace("\\", "/"))
        return selected[:6] or self._existing_key_files()

    def _planned_search_queries(self, evidence_plan: dict[str, Any] | None) -> list[str]:
        queries = [str(item) for item in (evidence_plan or {}).get("search_queries") or [] if str(item).strip()]
        return queries[:3] or ["class"]

    def _existing_key_files(self) -> list[str]:
        candidates = ["README.md", "src/main.py", "src/config.py", "src/runtime.py", "src/team.py"]
        return [path for path in candidates if Path(path).exists()][:5]

    def _extract_output_path(self, text: str) -> str | None:
        match = re.search(r"(outputs[\\/][\w./\\-]+\.(?:md|txt|py|json|csv))", text, re.I)
        return match.group(1).replace("\\", "/") if match else None

    def _extract_write_content(self, text: str) -> str:
        path = self._extract_output_path(text)
        if path:
            before_path = text.split(path, 1)[0].strip(" ：:，,。")
            for marker in ("把", "将", "写入", "写到", "输出到", "保存到", "保存为", "write", "save"):
                if marker in before_path:
                    content = before_path.split(marker, 1)[-1].strip(" ：:，,。")
                    if content:
                        return content
        return f"Team build output\n\n{text}"

    def _with_evidence_summary(self, description: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return description
        lines = [description, "", "Evidence Cards:"]
        lines.extend(self._evidence_cards(evidence))
        return "\n".join(lines)

    def _evidence_cards(self, evidence: list[dict[str, Any]]) -> list[str]:
        cards: list[str] = []
        for index, item in enumerate(evidence, start=1):
            evidence_id = f"ev_{index}"
            cards.append(f"- {evidence_id}")
            cards.append(f"  tool: {item.get('tool')}")
            cards.append(f"  status: {item.get('status')}")
            cards.append(f"  args: {item.get('args')}")
            cards.append(f"  summary: {self._evidence_summary_line(item)}")
            if item.get("raw_output_path"):
                cards.append(f"  artifact: {item.get('raw_output_path')}")
            if item.get("error"):
                cards.append(f"  error: {item.get('error')}")
        return cards

    def _evidence_summary_line(self, item: dict[str, Any]) -> str:
        if item.get("error"):
            return "tool call failed"
        tool = item.get("tool")
        metadata = item.get("metadata") or {}
        if tool == "list_files":
            return f"collected project file listing with {metadata.get('count', 'unknown')} entries"
        if tool == "read_file":
            return f"read key file {metadata.get('path') or item.get('args', {}).get('path')}"
        if tool == "search_files":
            return f"searched files for {item.get('args', {}).get('query')} with {metadata.get('count', 'unknown')} matches"
        return str(item.get("output") or "")[:240].replace("\n", " ")

    def _team_evidence_summary(self, team_run_id: str) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        for message in self.storage.list_team_messages(team_run_id):
            payload = message.get("payload") or {}
            if payload.get("type") == "evidence":
                evidence.append(
                    {
                        "tool": payload.get("tool"),
                        "args": payload.get("args"),
                        "status": payload.get("status"),
                        "output": payload.get("output"),
                        "error": payload.get("error"),
                        "metadata": payload.get("metadata") or {},
                        "raw_output_path": payload.get("raw_output_path"),
                    }
                )
        return evidence[:8]

    def _evidence_member_answer(self, item: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        lines = [
            f"Task: {item['title']}",
            "Evidence Cards:",
        ]
        lines.extend(self._evidence_cards(evidence))
        lines.extend(["", "Findings:"])
        for index, evidence_item in enumerate(evidence, start=1):
            if evidence_item.get("status") == "success":
                lines.append(f"- ev_{index}: {self._evidence_summary_line(evidence_item)}")
        lines.append("Assumptions: []")
        lines.append("Risks: []")
        lines.append("Next steps: []")
        return "\n".join(lines)

    def _member_session_id(self, team_run_id: str, member_name: str) -> str:
        for member in self.storage.list_team_members(team_run_id):
            if member["name"] == member_name:
                return member["session_id"]
        return new_session_id()

    def _agent_runtime(self, config: AppConfig):
        if self.agent_runtime_factory:
            return self.agent_runtime_factory(config, self.llm, self.storage)
        from src.runtime import AgentRuntime

        return AgentRuntime(config, llm=self.llm, storage=self.storage)

    def _summarize(self, task: str, spec: TeamSpec, member_results: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        reports = [MemberReport(**item["member_report"]) for item in member_results if item.get("member_report")]
        if reports:
            ContextHarness(
                self.storage,
                self.config.context_harness,
                Path(self.config.output_dir) / "context",
            ).join(reports)
        synthesizer = LeadSynthesizer()
        synthesis = synthesizer.synthesize(task, spec.name, member_results)
        synthesis_llm = self.llm if self._should_use_llm_synthesis(task, member_results) else None
        final_answer, synthesis_notes = synthesizer.render_with_optional_llm(task, spec.name, synthesis, member_results, synthesis_llm)
        learnings = {"conventions": [], "successes": [], "failures": [], "gotchas": [], "commands": [], "open_questions": []}
        for item in member_results:
            evidence_check = item.get("evidence_check") or {}
            if item.get("status") == "completed" and evidence_check.get("passed", True):
                learnings["successes"].append(f"{item['member']} completed {item['role']} work.")
            else:
                learnings["failures"].append(f"{item['member']} ended with status {item.get('status')}.")
            learnings["gotchas"].extend(evidence_check.get("issues") or [])
            issues = item.get("acceptance", {}).get("issues") if isinstance(item.get("acceptance"), dict) else None
            if issues:
                learnings["gotchas"].extend(issues)
        learnings["gotchas"].extend(synthesis.issues)
        learnings["gotchas"].extend(note for note in synthesis_notes if "fallback" in note or "error" in note)
        learnings["conventions"].append("Final answers must include project analysis, recommendations, priority route, and evidence ids.")
        learnings["conventions"].append("LLM synthesis is allowed only when its candidate answer passes the evidence gate.")
        if not learnings["open_questions"]:
            learnings["open_questions"].append("Review whether future runs should connect real external tools or shell execution.")
        return final_answer, learnings

    def _should_use_llm_synthesis(self, task: str, member_results: list[dict[str, Any]]) -> bool:
        if self.llm is None or not self.config.team_mode.llm_synthesis_enabled:
            return False
        text = task.lower()
        if any(keyword in text for keyword in ("claw-eval", "claw_eval", "agentbench", "deepeval", "benchmark", "评测", "测评")):
            return False
        for item in member_results:
            strategy = str(item.get("assignment_strategy") or "")
            metadata = item.get("task_metadata") if isinstance(item.get("task_metadata"), dict) else {}
            if strategy == "claw_baseline" or metadata.get("assignment_strategy") == "claw_baseline":
                return False
        return True

    def _evidence_findings(self, member_results: list[dict[str, Any]]) -> list[str]:
        findings: list[str] = []
        for result in member_results:
            for item in result.get("evidence") or []:
                if item.get("status") != "success":
                    continue
                tool = item.get("tool")
                metadata = item.get("metadata") or {}
                if tool == "list_files":
                    findings.append(f"- Project file listing collected with {metadata.get('count', 'unknown')} entries; raw output may be truncated.")
                elif tool == "read_file":
                    findings.append(f"- Read key file: {metadata.get('path') or item.get('args', {}).get('path')}.")
                elif tool == "search_files":
                    findings.append(f"- Searched source files for `{item.get('args', {}).get('query')}` and found {metadata.get('count', 'unknown')} matches.")
        return findings or ["- No tool-backed evidence was collected."]

    def _result(
        self,
        team_run_id: str,
        team_id: str,
        session_id: str,
        final_answer: str,
        learnings: dict[str, Any],
        status: str = "completed",
    ) -> TeamRunResult:
        return TeamRunResult(
            team_run_id=team_run_id,
            team_id=team_id,
            session_id=session_id,
            status=status,
            final_answer=final_answer,
            events=self.storage.list_events(session_id),
            members=self.storage.list_team_members(team_run_id),
            tasks=self.storage.list_team_tasks(team_run_id),
            messages=self.storage.list_team_messages(team_run_id),
            member_runs=self.storage.list_team_member_runs(team_run_id),
            learnings=learnings,
        )
