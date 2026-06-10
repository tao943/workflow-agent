from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import AgentSpec, AppConfig, TeamSpec
from src.project_files import ProjectFileFilter
from src.prompts.team_prompt import TEAM_LEAD_SYSTEM_PROMPT
from src.tool_intent import required_tools_for_task


TaskIntent = Literal["analysis", "planning", "research", "build", "review", "mixed"]
AnalysisSubtype = Literal["structure", "quality", "architecture", "testing", "engineering", "benchmark", "security", "performance"]
AnalysisDepth = Literal["quick", "standard", "deep"]
RiskLevel = Literal["low", "medium", "high"]
NodeType = Literal["task", "join", "review", "repair", "final"]
TaskCondition = Literal["always", "on_success", "on_failure", "if_evidence_missing", "if_user_requested_write"]
ReplanPolicy = Literal["never", "after_failure", "after_layer", "before_final"]


@dataclass
class MemberCapability:
    name: str
    role: str
    profile: str
    allowed_tools: set[str] = field(default_factory=set)
    can_write: bool = False
    can_delegate: bool = False
    evidence_strength: int = 0
    cost_level: str = "medium"
    parallel_safe: bool = True


@dataclass
class TaskNode:
    id: str
    title: str
    description: str
    assigned_to: str
    depends_on: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    acceptance_criteria: str = ""
    risk_level: RiskLevel = "low"
    assignment_reason: str = ""
    node_type: NodeType = "task"
    optional: bool = False
    condition: TaskCondition = "always"
    max_retries: int = 0
    replan_policy: ReplanPolicy = "never"
    why_this_member: str = ""
    why_now: str = ""
    skip_condition: str = ""
    replan_round: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGraph:
    intent: TaskIntent
    strategy: str
    tasks: list[TaskNode]
    issues: list[str] = field(default_factory=list)
    fallback_used: bool = False
    confidence: float = 1.0
    rationale_summary: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "strategy": self.strategy,
            "fallback_used": self.fallback_used,
            "confidence": self.confidence,
            "rationale_summary": self.rationale_summary,
            "issues": self.issues,
            "tasks": [task.__dict__ for task in self.tasks],
        }


@dataclass
class RouteDecision:
    intent: TaskIntent
    confidence: float
    rationale_summary: str
    required_capabilities: list[str]
    suggested_graph: TaskGraph
    fallback_used: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class AnalysisStrategy:
    analysis_subtype: AnalysisSubtype
    depth: AnalysisDepth
    focus_areas: list[str]
    required_evidence: list[str]
    suggested_files: list[str]
    search_queries: list[str]
    excluded_tools: list[str] = field(default_factory=list)
    strategy_reason: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "analysis_subtype": self.analysis_subtype,
            "depth": self.depth,
            "focus_areas": self.focus_areas,
            "required_evidence": self.required_evidence,
            "suggested_files": self.suggested_files,
            "search_queries": self.search_queries,
            "excluded_tools": self.excluded_tools,
            "strategy_reason": self.strategy_reason,
        }


class TaskClassifier:
    def classify(self, task: str) -> TaskIntent:
        text = task.lower()
        if any(keyword in text for keyword in ("写入", "保存到", "保存为", "outputs/", "write", "save")):
            return "build"
        has_build = any(keyword in text for keyword in ("实现", "修改", "修复", "优化代码", "加上", "新增", "implement", "fix", "build", "edit"))
        has_research = any(keyword in text for keyword in ("检索", "知识库", "rag", "搜索资料", "research", "lookup"))
        has_analysis = any(keyword in text for keyword in ("分析", "结构", "代码审查", "review code", "project structure", "analyze"))
        has_planning = any(keyword in text for keyword in ("计划", "规划", "方案", "plan", "roadmap"))
        has_review = any(keyword in text for keyword in ("验收", "检查", "review", "风险"))

        matches = [has_build, has_research, has_analysis, has_planning, has_review]
        if has_build and (has_analysis or has_planning or has_research):
            return "mixed"
        if has_build:
            return "build"
        if has_research:
            return "research"
        if has_analysis:
            return "analysis"
        if has_review:
            return "review"
        if has_planning:
            return "planning"
        if sum(1 for item in matches if item) > 1:
            return "mixed"
        return "planning"


class TaskClassifier:
    def classify(self, task: str) -> TaskIntent:
        text = task.lower()
        if any(keyword in text for keyword in ("写入", "保存到", "保存为", "输出到", "outputs/", "write", "save")):
            return "build"
        has_build = any(keyword in text for keyword in ("实现", "修改", "修复", "优化代码", "加上", "新增", "落地", "implement", "fix", "build", "edit"))
        has_research = any(keyword in text for keyword in ("检索", "知识库", "rag", "搜索资料", "research", "lookup"))
        has_analysis = any(keyword in text for keyword in ("分析", "结构", "项目问题", "代码审查", "review code", "project structure", "analyze", "benchmark"))
        has_planning = any(keyword in text for keyword in ("计划", "规划", "方案", "路线", "plan", "roadmap"))
        has_review = any(keyword in text for keyword in ("验收", "检查", "review", "风险"))

        if has_build and (has_analysis or has_planning or has_research):
            return "mixed"
        if has_build:
            return "build"
        if has_research:
            return "research"
        if has_analysis:
            return "analysis"
        if has_review:
            return "review"
        if has_planning:
            return "planning"
        return "planning"


class AnalysisStrategyRouter:
    def route(self, task: str) -> AnalysisStrategy:
        text = task.lower()
        depth: AnalysisDepth = "standard"
        if any(keyword in text for keyword in ("快速", "简单", "quick", "brief")):
            depth = "quick"
        if any(keyword in text for keyword in ("深入", "全面", "系统", "deep", "comprehensive")):
            depth = "deep"

        if any(keyword in text for keyword in ("测试", "pytest", "覆盖", "test", "coverage")):
            return ProjectEvidencePlanner().strategy("testing", depth, "用户关注测试体系、覆盖缺口或稳定性。")
        if any(keyword in text for keyword in ("benchmark", "deepeval", "claw", "agentbench", "测评", "评测", "低分")):
            return ProjectEvidencePlanner().strategy("benchmark", depth, "用户关注 benchmark 或 eval 结果。")
        if any(keyword in text for keyword in ("安全", "权限", "沙箱", "sandbox", "permission", "security")):
            return ProjectEvidencePlanner().strategy("security", depth, "用户关注权限、安全边界或沙箱。")
        if any(keyword in text for keyword in ("性能", "速度", "慢", "latency", "performance")):
            return ProjectEvidencePlanner().strategy("performance", depth, "用户关注速度、性能或延迟。")
        if any(keyword in text for keyword in ("工程化", "runtime", "toolprovider", "mcp", "skill", "memory", "context", "架构边界")):
            return ProjectEvidencePlanner().strategy("engineering", depth, "用户关注工程化边界和运行时模块。")
        if any(keyword in text for keyword in ("质量", "代码质量", "坏味道", "维护性", "quality", "maintainability")):
            return ProjectEvidencePlanner().strategy("quality", depth, "用户关注代码质量和可维护性。")
        if any(keyword in text for keyword in ("架构", "边界", "模块", "architecture")):
            return ProjectEvidencePlanner().strategy("architecture", depth, "用户关注架构和模块边界。")
        return ProjectEvidencePlanner().strategy("structure", depth, "默认项目结构分析策略。")


class ProjectEvidencePlanner:
    def strategy(self, subtype: AnalysisSubtype, depth: AnalysisDepth, reason: str) -> AnalysisStrategy:
        file_map: dict[str, list[str]] = {
            "structure": ["README.md", "src/main.py", "src/config.py", "src/runtime.py", "src/team.py"],
            "quality": ["src/runtime.py", "src/team.py", "src/tool_runtime.py", "src/storage.py", "src/assignment.py"],
            "architecture": ["src/runtime.py", "src/team.py", "src/storage.py", "src/context_harness.py", "src/memory_runtime.py"],
            "testing": ["pytest.ini", "tests/test_graph.py", "tests/test_assignment.py", "tests/test_evals.py", "tests/test_claw_workflow.py"],
            "engineering": ["src/runtime.py", "src/team.py", "src/tool_runtime.py", "src/config.py", "src/assignment.py"],
            "benchmark": ["src/evals.py", "src/benchmarks.py", "src/benchmark_protocols.py", "src/claw_workflow.py", "evals/benchmarks/baseline_v1.jsonl"],
            "security": ["src/permissions.py", "src/tool_runtime.py", "src/skill_plugins.py", "src/mcp_runtime.py", "src/claw_workflow.py"],
            "performance": ["src/runtime.py", "src/context_harness.py", "src/team.py", "src/evals.py", "src/mcp_runtime.py"],
        }
        query_map: dict[str, list[str]] = {
            "structure": ["class", "def main|argparse|TeamRuntime"],
            "quality": ["TODO|FIXME|except Exception|pass", "class .*Runtime|class .*Manager"],
            "architecture": ["class AgentRuntime|class TeamRuntime|class Storage", "ContextHarness|MemoryGovernanceRuntime"],
            "testing": ["skip|xfail|pytest.mark", "RUN_REAL_|integration|e2e"],
            "engineering": ["ToolRuntime|Permission|ContextHarness|MemoryGovernanceRuntime", "class .*Provider|MCP|Skill"],
            "benchmark": ["DeepEval|Benchmark|Claw|AgentBench", "native_tool_success|adapter_repaired|task_score"],
            "security": ["Permission|deny|ask|sandbox|network|shell", "docker|no-new-privileges|cap-drop"],
            "performance": ["cache|timeout|duration|ThreadPoolExecutor|parallel", "prompt_cache|ContextBundle"],
        }
        files = self._existing_filtered(file_map[subtype])
        if depth == "quick":
            files = files[:2]
            queries = query_map[subtype][:1]
        elif depth == "deep":
            queries = query_map[subtype]
        else:
            files = files[:4]
            queries = query_map[subtype][:1]
        required_evidence = [
            f"{subtype} file evidence",
            f"{subtype} search evidence",
        ]
        return AnalysisStrategy(
            analysis_subtype=subtype,
            depth=depth,
            focus_areas=[subtype],
            required_evidence=required_evidence,
            suggested_files=files,
            search_queries=queries,
            excluded_tools=["write_file", "notes_tool"],
            strategy_reason=reason,
        )

    def _existing_filtered(self, candidates: list[str]) -> list[str]:
        project_filter = ProjectFileFilter()
        result: list[str] = []
        for candidate in candidates:
            path = project_filter.workspace / candidate
            if path.exists() and project_filter.should_include(path):
                result.append(candidate)
        return result


class SemanticTaskRouter:
    def __init__(self, config: AppConfig, llm: Any | None = None) -> None:
        self.config = config
        self.llm = llm
        self.classifier = TaskClassifier()

    def route(
        self,
        task: str,
        spec: TeamSpec,
        capabilities: dict[str, MemberCapability],
        baseline: TaskGraph,
    ) -> RouteDecision:
        if not self.llm:
            return RouteDecision(
                intent=baseline.intent,
                confidence=0.0,
                rationale_summary="no llm; using baseline",
                required_capabilities=[],
                suggested_graph=baseline,
                fallback_used=True,
                issues=["semantic_router.no_llm"],
            )
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=TEAM_LEAD_SYSTEM_PROMPT),
                    HumanMessage(content=build_semantic_route_prompt(task, spec, capabilities, baseline)),
                ]
            )
            payload = json.loads(str(getattr(response, "content", response) or ""))
            return route_decision_from_payload(payload, baseline)
        except Exception as exc:
            return RouteDecision(
                intent=baseline.intent,
                confidence=0.0,
                rationale_summary="semantic route failed; using baseline",
                required_capabilities=[],
                suggested_graph=baseline,
                fallback_used=True,
                issues=[f"semantic_router.error:{exc}"],
            )


class AssignmentPlanner:
    def __init__(self, config: AppConfig, llm: Any | None = None) -> None:
        self.config = config
        self.llm = llm
        self.classifier = TaskClassifier()

    def plan(self, task: str, spec: TeamSpec) -> TaskGraph:
        intent = self.classifier.classify(task)
        capabilities = build_member_capabilities(spec)
        if is_claw_task(task):
            claw_graph = build_claw_task_graph(task, spec)
            issues = GraphGate(self.config, capabilities).validate(claw_graph)
            claw_graph.issues.extend(issues)
            return claw_graph
        baseline = self.build_baseline_task_graph(task, spec, intent, capabilities)
        gate = GraphGate(self.config, capabilities)
        baseline_issues = gate.validate(baseline)
        if baseline_issues:
            baseline.issues.extend(baseline_issues)
            return baseline

        strategy = self.config.team_mode.assignment_strategy
        if (
            strategy == "rules"
            or not self.llm
            or not self.config.team_mode.llm_assignment_enabled
            or not self.config.team_mode.semantic_routing_enabled
        ):
            return baseline

        route = SemanticTaskRouter(self.config, self.llm).route(task, spec, capabilities, baseline)
        if route.fallback_used or route.confidence < self.config.team_mode.min_route_confidence:
            baseline.fallback_used = route.fallback_used
            baseline.issues.extend(route.issues)
            return baseline
        refined = route.suggested_graph
        issues = gate.validate(refined)
        if issues:
            baseline.issues.extend([f"llm_assignment.fallback:{issue}" for issue in issues])
            baseline.fallback_used = True
            baseline.strategy = "fallback"
            return baseline
        return refined

    def build_baseline_task_graph(
        self,
        task: str,
        spec: TeamSpec,
        intent: TaskIntent,
        capabilities: dict[str, MemberCapability],
    ) -> TaskGraph:
        tasks: list[TaskNode] = []
        researcher = find_member(spec, "researcher")
        planner = find_member(spec, "planner")
        builder = find_member(spec, "builder")
        reviewer = find_member(spec, "reviewer")

        def add(node: TaskNode) -> None:
            if len(tasks) < self.config.team_mode.max_assignment_tasks:
                tasks.append(node)

        if intent == "analysis":
            strategy = AnalysisStrategyRouter().route(task)
            tasks.extend(build_analysis_task_nodes(task, strategy, researcher, planner, builder, reviewer, self.config.team_mode.max_assignment_tasks))
        elif intent == "research":
            if researcher:
                add(
                    TaskNode(
                        id="rag_research",
                        title="Search knowledge base",
                        description=member_task_description(researcher, task, ["RAG sources or not_configured error"]),
                        assigned_to=researcher.name,
                        required_tools=["rag_search"],
                        required_evidence=["RAG search sources or explicit not_configured result"],
                        acceptance_criteria="RAG result or not_configured status is recorded as tool evidence.",
                        assignment_reason="research intent should start with RAG evidence",
                    )
                )
            if reviewer:
                add(
                    TaskNode(
                        id="review_research",
                        title="Review research sources",
                        description=member_task_description(reviewer, task, ["RAG evidence"]),
                        assigned_to=reviewer.name,
                        depends_on=["rag_research"] if researcher else [],
                        required_evidence=["RAG evidence"],
                        acceptance_criteria="Reviewer verifies that claims cite RAG evidence.",
                        assignment_reason="research output needs source validation",
                        node_type="review",
                    )
                )
        elif intent == "build" or intent == "mixed":
            if researcher:
                add(project_evidence_task(researcher))
            if planner:
                add(
                    TaskNode(
                        id="plan_build",
                        title="Plan implementation steps",
                        description=member_task_description(planner, task, ["project evidence"]),
                        assigned_to=planner.name,
                        depends_on=["collect_project_evidence"] if researcher else [],
                        required_evidence=["project evidence"],
                        acceptance_criteria="Plan includes executable steps and acceptance criteria.",
                        assignment_reason="build tasks need a plan before file changes",
                    )
                )
            if builder:
                build_tools = required_tools_for_task(task, "build")
                if any(tool in build_tools for tool in ("write_file", "notes_tool")) and "read_file" not in build_tools:
                    build_tools.insert(0, "read_file")
                build_tools = [tool for tool in build_tools if tool in capability_from_member(builder).allowed_tools]
                add(
                    TaskNode(
                        id="execute_build",
                        title="Execute build-oriented work",
                        description=member_task_description(builder, task, ["plan and project evidence"]),
                        assigned_to=builder.name,
                        depends_on=[item.id for item in tasks],
                        required_tools=build_tools,
                        required_evidence=["plan and project evidence"],
                        acceptance_criteria="Requested implementation work is completed or a blocker is reported.",
                        risk_level="medium",
                        assignment_reason="user intent requires implementation-capable member",
                    )
                )
            if reviewer:
                add(
                    TaskNode(
                        id="review_build",
                        title="Review build results",
                        description=member_task_description(reviewer, task, ["build output and evidence"]),
                        assigned_to=reviewer.name,
                        depends_on=[item.id for item in tasks],
                        required_evidence=["build output and evidence"],
                        acceptance_criteria="Reviewer checks completion, risks, and missing tests.",
                        assignment_reason="build work requires final validation",
                        node_type="review",
                    )
                )
        elif intent == "review":
            if reviewer:
                add(
                    TaskNode(
                        id="review_outputs",
                        title="Review outputs and risks",
                        description=member_task_description(reviewer, task, ["provided context"]),
                        assigned_to=reviewer.name,
                        required_evidence=["provided context"],
                        acceptance_criteria="Review lists findings, risks, and missing evidence.",
                        assignment_reason="review intent should route directly to reviewer",
                        node_type="review",
                    )
                )
        else:
            if planner:
                add(
                    TaskNode(
                        id="create_plan",
                        title="Create implementation plan",
                        description=member_task_description(planner, task, ["user task"]),
                        assigned_to=planner.name,
                        required_evidence=["user task"],
                        acceptance_criteria="Plan is specific and testable.",
                        assignment_reason="planning intent should route to planner first",
                    )
                )
            if reviewer:
                add(
                    TaskNode(
                        id="review_plan",
                        title="Review plan quality",
                        description=member_task_description(reviewer, task, ["planner output"]),
                        assigned_to=reviewer.name,
                        depends_on=["create_plan"] if planner else [],
                        required_evidence=["planner output"],
                        acceptance_criteria="Reviewer flags gaps and vague acceptance criteria.",
                        assignment_reason="plans should be checked before synthesis",
                        node_type="review",
                    )
                )

        return TaskGraph(intent=intent, strategy="baseline", tasks=tasks, rationale_summary="rule baseline")

    def _llm_refine(
        self,
        task: str,
        spec: TeamSpec,
        intent: TaskIntent,
        capabilities: dict[str, MemberCapability],
        baseline: TaskGraph,
    ) -> TaskGraph | None:
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=TEAM_LEAD_SYSTEM_PROMPT),
                    HumanMessage(content=build_assignment_prompt(task, spec, intent, capabilities, baseline)),
                ]
            )
            payload = json.loads(str(getattr(response, "content", response) or ""))
            selected = payload.get("selected") or payload
            return task_graph_from_payload(intent, selected, strategy="llm_refined")
        except Exception:
            if self.config.team_mode.assignment_strategy == "llm":
                baseline.fallback_used = True
                baseline.strategy = "fallback"
            return None


class GraphGate:
    def __init__(self, config: AppConfig, capabilities: dict[str, MemberCapability]) -> None:
        self.config = config
        self.capabilities = capabilities

    def validate(self, graph: TaskGraph) -> list[str]:
        issues: list[str] = []
        if not graph.tasks:
            issues.append("task graph is empty")
            return issues
        if len(graph.tasks) > self.config.team_mode.max_assignment_tasks:
            issues.append("task graph exceeds max_assignment_tasks")

        task_ids = {task.id for task in graph.tasks}
        issues.extend(self._cycle_issues(graph))
        for task in graph.tasks:
            capability = self.capabilities.get(task.assigned_to)
            if not capability:
                issues.append(f"assigned member does not exist: {task.assigned_to}")
                continue
            missing_deps = [dep for dep in task.depends_on if dep not in task_ids]
            if missing_deps:
                issues.append(f"task {task.id} depends on missing tasks: {missing_deps}")
            missing_tools = [tool for tool in task.required_tools if tool not in capability.allowed_tools]
            if missing_tools:
                issues.append(f"task {task.id} uses tools not allowed for {task.assigned_to}: {missing_tools}")
            if any(tool in {"write_file", "notes_tool"} for tool in task.required_tools) and not capability.can_write:
                issues.append(f"write task {task.id} assigned to read-only member {task.assigned_to}")
            if task.node_type in {"review", "final"} and not task.depends_on and len(graph.tasks) > 1:
                issues.append(f"{task.node_type} task {task.id} must depend on prior work")

        evidence_tasks = {task.id for task in graph.tasks if task.required_tools or "evidence" in " ".join(task.required_evidence).lower()}
        for task in graph.tasks:
            if task.risk_level in {"medium", "high"} and not task.depends_on and task.required_tools:
                continue
            if task.assigned_to in self.capabilities and self.capabilities[task.assigned_to].role in {"planner", "builder", "reviewer"}:
                needs_facts = any(word in task.description.lower() for word in ("project", "code", "file", "evidence", "项目", "代码", "文件", "证据"))
                if needs_facts and evidence_tasks and not set(task.depends_on).intersection(evidence_tasks) and task.id not in evidence_tasks:
                    issues.append(f"task {task.id} needs factual evidence dependency")

        execution_tasks = [task.id for task in graph.tasks if self.capabilities.get(task.assigned_to, None) and self.capabilities[task.assigned_to].role in {"builder", "planner", "researcher"}]
        reviewer_tasks = [task for task in graph.tasks if self.capabilities.get(task.assigned_to, None) and self.capabilities[task.assigned_to].role == "reviewer"]
        for reviewer in reviewer_tasks:
            missing = [task_id for task_id in execution_tasks if task_id != reviewer.id and task_id not in reviewer.depends_on]
            if missing:
                issues.append(f"reviewer task {reviewer.id} does not depend on prior work: {missing}")
        return issues

    def _cycle_issues(self, graph: TaskGraph) -> list[str]:
        graph_edges = {task.id: list(task.depends_on) for task in graph.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for dep in graph_edges.get(node_id, []):
                if dep in graph_edges and visit(dep):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return ["task graph has circular dependency"] if any(visit(node_id) for node_id in graph_edges) else []


AssignmentGate = GraphGate


class GraphReplanner:
    def __init__(self, config: AppConfig, capabilities: dict[str, MemberCapability]) -> None:
        self.config = config
        self.capabilities = capabilities

    def maybe_replan(
        self,
        graph: TaskGraph,
        completed: set[str],
        failed: set[str],
        results: list[dict[str, Any]],
        replan_round: int,
    ) -> list[TaskNode]:
        if replan_round >= self.config.team_mode.max_replans_per_run:
            return []
        if self.config.team_mode.replan_strategy == "before_final" and not self._has_final_ready(graph, completed):
            return []

        existing_ids = {task.id for task in graph.tasks}
        additions: list[TaskNode] = []
        if failed and len(additions) < self.config.team_mode.max_repair_tasks:
            reviewer = self._member_by_role("reviewer")
            if reviewer and f"repair_review_{replan_round + 1}" not in existing_ids:
                additions.append(
                    TaskNode(
                        id=f"repair_review_{replan_round + 1}",
                        title="Review failed task and propose repair",
                        description="Review failed team task results and propose the smallest safe repair path.",
                        assigned_to=reviewer.name,
                        depends_on=sorted(completed),
                        required_evidence=["failed task result"],
                        acceptance_criteria="Repair review lists blocker, impact, and next action.",
                        node_type="repair",
                        condition="on_failure",
                        assignment_reason="a dependency failed and needs bounded repair analysis",
                        replan_round=replan_round + 1,
                    )
                )

        missing_evidence = any(
            item.get("evidence_check", {}).get("status") in {"missing", "partial"}
            or item.get("evidence_check", {}).get("passed") is False
            for item in results
        )
        if missing_evidence and len(additions) < self.config.team_mode.max_repair_tasks:
            researcher = self._member_by_role("researcher")
            task_id = f"supplement_evidence_{replan_round + 1}"
            if researcher and task_id not in existing_ids:
                additions.append(
                    TaskNode(
                        id=task_id,
                        title="Supplement missing evidence",
                        description="Collect focused additional evidence for unresolved claims or missing tool outputs.",
                        assigned_to=researcher.name,
                        depends_on=sorted(completed),
                        required_tools=["list_files", "search_files"],
                        required_evidence=["supplemental evidence"],
                        acceptance_criteria="Supplemental tool evidence is recorded or the blocker is explicit.",
                        node_type="repair",
                        condition="if_evidence_missing",
                        assignment_reason="reviewer or evidence gate reported missing evidence",
                        replan_round=replan_round + 1,
                    )
                )
        return additions[: self.config.team_mode.max_repair_tasks]

    def should_skip(self, task: TaskNode, user_task: str, completed: set[str], failed: set[str]) -> str | None:
        if task.condition == "always":
            return None
        if task.condition == "on_success" and not set(task.depends_on).issubset(completed):
            return "condition on_success not satisfied"
        if task.condition == "on_failure" and not set(task.depends_on).intersection(failed):
            return "condition on_failure not satisfied"
        if task.condition == "if_user_requested_write":
            text = user_task.lower()
            if not any(keyword in text for keyword in ("写", "保存", "修改", "实现", "write", "save", "edit", "implement")):
                return "user did not request write-capable work"
        return None

    def _member_by_role(self, role: str) -> MemberCapability | None:
        return next((item for item in self.capabilities.values() if item.role == role), None)

    def _has_final_ready(self, graph: TaskGraph, completed: set[str]) -> bool:
        final_nodes = [task for task in graph.tasks if task.node_type == "final"]
        return any(set(task.depends_on).issubset(completed) for task in final_nodes)


def build_member_capabilities(spec: TeamSpec) -> dict[str, MemberCapability]:
    return {member.name: capability_from_member(member) for member in spec.members}


def capability_from_member(member: AgentSpec) -> MemberCapability:
    default_tools = {
        "researcher": {"list_files", "read_file", "search_files", "rag_search"},
        "planner": set(),
        "builder": {"read_file", "write_file", "list_files", "search_files", "notes_tool"},
        "reviewer": set(),
    }
    allowed_tools = set(member.allowed_tools or default_tools.get(member.role, set()))
    return MemberCapability(
        name=member.name,
        role=member.role,
        profile=member.profile,
        allowed_tools=allowed_tools,
        can_write=member.can_write or member.profile == "build",
        can_delegate=member.can_delegate,
        evidence_strength=3 if member.role == "researcher" else 1,
        cost_level=member.cost_level,
        parallel_safe=member.role not in {"lead"},
    )


def is_claw_task(task: str) -> bool:
    return "Claw-Eval task:" in task and "Available Claw tools:" in task


def build_analysis_task_nodes(
    task: str,
    strategy: AnalysisStrategy,
    researcher: AgentSpec | None,
    planner: AgentSpec | None,
    builder: AgentSpec | None,
    reviewer: AgentSpec | None,
    max_tasks: int,
) -> list[TaskNode]:
    nodes: list[TaskNode] = []
    metadata = {"analysis_strategy": strategy.model_dump()}

    def add(node: TaskNode) -> None:
        if len(nodes) < max_tasks:
            nodes.append(node)

    evidence_ids: list[str] = []
    if researcher:
        if strategy.depth == "deep" and len(strategy.focus_areas) > 1:
            previous: str | None = None
            for index, focus in enumerate(strategy.focus_areas, start=1):
                node_id = f"collect_{focus}_evidence"
                evidence_ids.append(node_id)
                add(
                    TaskNode(
                        id=node_id,
                        title=f"Collect {focus} evidence",
                        description=member_task_description(researcher, task, strategy.required_evidence),
                        assigned_to=researcher.name,
                        depends_on=[previous] if previous else [],
                        required_tools=["list_files", "read_file", "search_files"],
                        required_evidence=strategy.required_evidence,
                        acceptance_criteria="Focused project evidence is recorded with tool calls.",
                        assignment_reason=strategy.strategy_reason,
                        metadata={**metadata, "focus_area": focus, "evidence_plan": strategy.model_dump()},
                    )
                )
                previous = node_id
        else:
            evidence_ids.append("collect_project_evidence")
            add(project_evidence_task(researcher, strategy=strategy, user_task=task))

    if strategy.depth != "quick" and planner:
        add(
            TaskNode(
                id="plan_analysis",
                title=f"Analyze {strategy.analysis_subtype} evidence",
                description=member_task_description(planner, task, strategy.required_evidence),
                assigned_to=planner.name,
                depends_on=evidence_ids,
                required_evidence=strategy.required_evidence,
                acceptance_criteria="Findings and recommendations cite evidence cards.",
                assignment_reason=f"analysis strategy selected {strategy.analysis_subtype}/{strategy.depth}",
                metadata={**metadata, "focus_area": ",".join(strategy.focus_areas), "evidence_plan": strategy.model_dump()},
            )
        )

    user_requested_build = any(keyword in task.lower() for keyword in ("实现", "修改", "修复", "落地", "优化代码", "implement", "fix", "edit"))
    if user_requested_build and builder:
        add(
            TaskNode(
                id="execute_analysis_followup",
                title="Execute requested analysis follow-up",
                description=member_task_description(builder, task, ["analysis plan and project evidence"]),
                assigned_to=builder.name,
                depends_on=[node.id for node in nodes],
                required_tools=[tool for tool in required_tools_for_task(task, "build") if tool in capability_from_member(builder).allowed_tools],
                required_evidence=["analysis plan and project evidence"],
                acceptance_criteria="Requested implementation follow-up is completed or a blocker is explicit.",
                risk_level="medium",
                assignment_reason="user asked analysis plus implementation work",
                metadata={**metadata, "focus_area": "implementation", "evidence_plan": strategy.model_dump()},
            )
        )

    if strategy.depth == "standard" and reviewer:
        add(
            TaskNode(
                id="review_analysis",
                title=f"Review {strategy.analysis_subtype} analysis",
                description=member_task_description(reviewer, task, ["all prior task outputs"]),
                assigned_to=reviewer.name,
                depends_on=[node.id for node in nodes],
                required_evidence=strategy.required_evidence,
                acceptance_criteria="Reviewer identifies unsupported claims and missing evidence.",
                assignment_reason="standard analysis should be evidence-gated before synthesis",
                node_type="review",
                metadata={**metadata, "focus_area": "review", "evidence_plan": strategy.model_dump()},
            )
        )
    elif strategy.depth == "deep" and reviewer:
        add(
            TaskNode(
                id="review_deep_analysis",
                title=f"Review deep {strategy.analysis_subtype} analysis",
                description=member_task_description(reviewer, task, ["all focused evidence branches"]),
                assigned_to=reviewer.name,
                depends_on=[node.id for node in nodes],
                required_evidence=strategy.required_evidence,
                acceptance_criteria="Reviewer checks focused evidence coverage and unresolved assumptions.",
                assignment_reason="deep analysis needs final evidence coverage review",
                node_type="review",
                metadata={**metadata, "focus_area": "review", "evidence_plan": strategy.model_dump()},
            )
        )
    return nodes


def claw_tool_names(task: str) -> list[str]:
    return re_find_tool_names(task)


def re_find_tool_names(task: str) -> list[str]:
    import re

    names: list[str] = []
    for match in re.finditer(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", task, re.M):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def build_claw_task_graph(task: str, spec: TeamSpec) -> TaskGraph:
    tools = set(claw_tool_names(task))
    researcher = find_member(spec, "researcher")
    planner = find_member(spec, "planner")
    builder = find_member(spec, "builder")
    reviewer = find_member(spec, "reviewer")
    tasks: list[TaskNode] = []

    def add(node: TaskNode) -> None:
        tasks.append(node)

    if {"kb_search", "kb_get_article"}.issubset(tools):
        if researcher:
            add(
                TaskNode(
                    id="claw_collect_kb_evidence",
                    title="Collect Claw KB evidence",
                    description=member_task_description(researcher, task, ["kb_search and kb_get_article evidence"]),
                    assigned_to=researcher.name,
                    required_tools=["kb_search", "kb_get_article"],
                    required_evidence=["KB search results and article details"],
                    acceptance_criteria="KB search and article reads are recorded; kb_update_article is not called.",
                    assignment_reason="Claw KB task requires benchmark HTTP tools, not project file tools.",
                )
            )
        if planner:
            add(
                TaskNode(
                    id="claw_synthesize_kb_answer",
                    title="Synthesize Claw KB answer",
                    description=member_task_description(planner, task, ["KB evidence"]),
                    assigned_to=planner.name,
                    depends_on=["claw_collect_kb_evidence"] if researcher else [],
                    required_evidence=["KB evidence"],
                    acceptance_criteria="Answer includes checklist, source ids, and conflict resolution.",
                    assignment_reason="Benchmark answer must be grounded in KB evidence.",
                )
            )
        if reviewer:
            add(
                TaskNode(
                    id="claw_review_kb_answer",
                    title="Review Claw KB answer",
                    description=member_task_description(reviewer, task, ["KB evidence and final answer"]),
                    assigned_to=reviewer.name,
                    depends_on=[node.id for node in tasks],
                    required_evidence=["KB evidence and final answer"],
                    acceptance_criteria="Reviewer confirms sources and absence of forbidden kb_update_article.",
                    assignment_reason="Claw grader rewards evidence-backed synthesis and forbidden-tool avoidance.",
                    node_type="review",
                )
            )
        return TaskGraph(intent="research", strategy="claw_baseline", tasks=tasks, rationale_summary="Claw KB task graph")

    if {"notes_list", "notes_get", "notes_share"}.issubset(tools):
        if researcher:
            add(
                TaskNode(
                    id="claw_collect_notes_evidence",
                    title="Collect Claw meeting notes",
                    description=member_task_description(researcher, task, ["notes_list and notes_get evidence"]),
                    assigned_to=researcher.name,
                    required_tools=["notes_list", "notes_get"],
                    required_evidence=["target meeting note and cross-referenced prior note"],
                    acceptance_criteria="note_001 and relevant work notes are retrieved; casual note is ignored.",
                    assignment_reason="Claw notes task requires meeting-note tools before sharing.",
                )
            )
        if builder:
            add(
                TaskNode(
                    id="claw_share_notes",
                    title="Share Claw meeting notes",
                    description=member_task_description(builder, task, ["meeting participants from note evidence"]),
                    assigned_to=builder.name,
                    depends_on=["claw_collect_notes_evidence"] if researcher else [],
                    required_tools=["notes_share"],
                    required_evidence=["successful notes_share action"],
                    acceptance_criteria="note_001 is shared only with meeting participants.",
                    risk_level="medium",
                    assignment_reason="The benchmark requires an external share action.",
                )
            )
        if reviewer:
            add(
                TaskNode(
                    id="claw_review_notes",
                    title="Review Claw meeting note answer",
                    description=member_task_description(reviewer, task, ["notes evidence and share action"]),
                    assigned_to=reviewer.name,
                    depends_on=[node.id for node in tasks],
                    required_evidence=["notes evidence and share action"],
                    acceptance_criteria="Reviewer checks action items, carry-over item, and recipients.",
                    assignment_reason="Claw notes grader checks extraction plus external action.",
                    node_type="review",
                )
            )
        return TaskGraph(intent="build", strategy="claw_baseline", tasks=tasks, rationale_summary="Claw notes task graph")

    if {"contacts_search", "contacts_get"}.issubset(tools):
        if researcher:
            add(
                TaskNode(
                    id="claw_collect_contact_evidence",
                    title="Collect Claw contact evidence",
                    description=member_task_description(researcher, task, ["contacts_search and contacts_get evidence"]),
                    assigned_to=researcher.name,
                    required_tools=["contacts_search", "contacts_get"],
                    required_evidence=["search matches and correct contact detail"],
                    acceptance_criteria="Search results are disambiguated and contacts_send_message is not called.",
                    assignment_reason="Claw contact task requires contact lookup tools and safety avoidance.",
                )
            )
        if reviewer:
            add(
                TaskNode(
                    id="claw_review_contact_answer",
                    title="Review Claw contact answer",
                    description=member_task_description(reviewer, task, ["contact lookup evidence"]),
                    assigned_to=reviewer.name,
                    depends_on=["claw_collect_contact_evidence"] if researcher else [],
                    required_evidence=["contact lookup evidence"],
                    acceptance_criteria="Reviewer checks exact-name recommendation and no message send.",
                    assignment_reason="Claw contact grader checks disambiguation and safety.",
                    node_type="review",
                )
            )
        return TaskGraph(intent="research", strategy="claw_baseline", tasks=tasks, rationale_summary="Claw contact task graph")

    return TaskGraph(intent="research", strategy="claw_baseline", tasks=tasks, issues=["unsupported Claw tool set"], rationale_summary="unsupported Claw task")


def find_member(spec: TeamSpec, role: str) -> AgentSpec | None:
    return next((member for member in spec.members if member.role == role), None)


def project_evidence_task(member: AgentSpec, strategy: AnalysisStrategy | None = None, user_task: str = "Analyze project context") -> TaskNode:
    evidence_plan = strategy.model_dump() if strategy else {
        "analysis_subtype": "structure",
        "depth": "standard",
        "focus_areas": ["structure"],
        "required_evidence": ["project file list", "key source files", "code search matches"],
        "suggested_files": ["README.md", "src/main.py", "src/config.py", "src/runtime.py", "src/team.py"],
        "search_queries": ["class"],
        "excluded_tools": ["write_file", "notes_tool"],
        "strategy_reason": "fallback project evidence plan",
    }
    return TaskNode(
        id="collect_project_evidence",
        title=f"Collect {evidence_plan['analysis_subtype']} evidence",
        description=member_task_description(member, user_task, list(evidence_plan["required_evidence"])),
        assigned_to=member.name,
        required_tools=["list_files", "read_file", "search_files"],
        required_evidence=list(evidence_plan["required_evidence"]),
        acceptance_criteria="Planned project file listing, key file reads, and focused source search are recorded as tool evidence.",
        assignment_reason=str(evidence_plan["strategy_reason"]),
        metadata={
            "analysis_subtype": evidence_plan["analysis_subtype"],
            "focus_area": ",".join(evidence_plan["focus_areas"]),
            "strategy_reason": evidence_plan["strategy_reason"],
            "evidence_plan": evidence_plan,
        },
    )


def member_task_description(member: AgentSpec, task: str, required_evidence: list[str]) -> str:
    return "\n".join(
        [
            member.system_prompt or f"Act as {member.role}.",
            f"User task: {task}",
            f"Required evidence: {required_evidence or 'none'}",
            "Do not claim scanned/read/searched/saved/written work unless matching evidence exists.",
            "Return concise findings, completed work, risks, and next steps.",
        ]
    )


def build_assignment_prompt(
    task: str,
    spec: TeamSpec,
    intent: TaskIntent,
    capabilities: dict[str, MemberCapability],
    baseline: TaskGraph,
) -> str:
    members = [
        {
            "name": item.name,
            "role": item.role,
            "profile": item.profile,
            "allowed_tools": sorted(item.allowed_tools),
            "can_write": item.can_write,
        }
        for item in capabilities.values()
    ]
    return json.dumps(
        {
            "user_task": task,
            "team": spec.name,
            "intent": intent,
            "candidate_members": members,
            "baseline": baseline.model_dump(),
            "instructions": "Return only JSON. You may reorder, remove unnecessary tasks, or add missing dependencies. Use only candidate members.",
        },
        ensure_ascii=False,
    )


def build_semantic_route_prompt(
    task: str,
    spec: TeamSpec,
    capabilities: dict[str, MemberCapability],
    baseline: TaskGraph,
) -> str:
    return json.dumps(
        {
            "user_task": task,
            "team": spec.name,
            "candidate_members": [
                {
                    "name": item.name,
                    "role": item.role,
                    "profile": item.profile,
                    "allowed_tools": sorted(item.allowed_tools),
                    "can_write": item.can_write,
                    "cost_level": item.cost_level,
                }
                for item in capabilities.values()
            ],
            "baseline_graph": baseline.model_dump(),
            "output_contract": {
                "intent": "analysis|planning|research|build|review|mixed",
                "confidence": "0.0-1.0",
                "rationale_summary": "short reason, no hidden chain of thought",
                "required_capabilities": [],
                "suggested_graph": {"tasks": []},
            },
            "rules": [
                "Return JSON only.",
                "Use only candidate member names.",
                "Do not claim tool results.",
                "Every task must include why_this_member, why_now, skip_condition, node_type, condition, and replan_policy.",
            ],
        },
        ensure_ascii=False,
    )


def route_decision_from_payload(payload: dict[str, Any], baseline: TaskGraph) -> RouteDecision:
    intent = payload.get("intent") if payload.get("intent") in {"analysis", "planning", "research", "build", "review", "mixed"} else baseline.intent
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    graph_payload = payload.get("suggested_graph") or payload.get("selected") or payload
    graph = task_graph_from_payload(intent, graph_payload, strategy="llm_semantic")
    graph.confidence = confidence
    graph.rationale_summary = str(payload.get("rationale_summary") or "")
    return RouteDecision(
        intent=intent,
        confidence=confidence,
        rationale_summary=graph.rationale_summary,
        required_capabilities=list(payload.get("required_capabilities") or []),
        suggested_graph=graph,
    )


def task_graph_from_payload(intent: TaskIntent, payload: dict[str, Any], strategy: str) -> TaskGraph:
    tasks_payload = payload.get("tasks") or []
    tasks: list[TaskNode] = []
    for index, item in enumerate(tasks_payload, start=1):
        tasks.append(
            TaskNode(
                id=str(item.get("id") or f"task_{index}"),
                title=str(item.get("title") or item.get("description") or f"Task {index}"),
                description=str(item.get("description") or item.get("title") or ""),
                assigned_to=str(item.get("assigned_to") or ""),
                depends_on=list(item.get("depends_on") or []),
                required_tools=list(item.get("required_tools") or []),
                required_evidence=list(item.get("required_evidence") or []),
                acceptance_criteria=str(item.get("acceptance_criteria") or item.get("done_criteria") or ""),
                risk_level=item.get("risk_level") if item.get("risk_level") in {"low", "medium", "high"} else "low",
                assignment_reason=str(item.get("assignment_reason") or "llm refined task graph"),
                node_type=item.get("node_type") if item.get("node_type") in {"task", "join", "review", "repair", "final"} else "task",
                optional=bool(item.get("optional", False)),
                condition=item.get("condition") if item.get("condition") in {"always", "on_success", "on_failure", "if_evidence_missing", "if_user_requested_write"} else "always",
                max_retries=int(item.get("max_retries", 0) or 0),
                replan_policy=item.get("replan_policy") if item.get("replan_policy") in {"never", "after_failure", "after_layer", "before_final"} else "never",
                why_this_member=str(item.get("why_this_member") or ""),
                why_now=str(item.get("why_now") or ""),
                skip_condition=str(item.get("skip_condition") or ""),
                replan_round=int(item.get("replan_round", 0) or 0),
            )
        )
    return TaskGraph(intent=intent, strategy=strategy, tasks=tasks)
