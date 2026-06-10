from dataclasses import dataclass, field
from typing import Any, Callable

from src.checkpoint import create_run_id, load_checkpoint, save_checkpoint
from src.config import AppConfig, permissions_for_agent
from src.context_manager import ContextBundle, ContextManager
from src.context_harness import ContextHarness, ContextRequest, ModelUsageRecorder
from src.graph import build_graph, make_initial_state
from src.integrations import IntegrationCatalog
from src.memory import load_memory, remember_interaction, save_memory
from src.memory_runtime import MemoryGovernanceRuntime
from src.session import create_or_continue_session, new_id, new_session_id
from src.storage import Storage


@dataclass
class RunResult:
    run_id: str
    session_id: str
    status: str
    final_answer: str
    acceptance: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    team: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeOptions:
    task: str | None = None
    agent: str | None = None
    run_id: str | None = None
    resume: str | None = None
    session_id: str | None = None
    continue_session: bool = False
    fork: bool = False
    auto_approve: bool = False
    output_format: str = "default"
    no_memory: bool = False
    full_model: bool = False
    team: str | None = None
    context_namespace: str | None = None
    consolidate_memory: bool = True


class AgentRuntime:
    def __init__(
        self,
        config: AppConfig,
        llm=None,
        graph_builder: Callable = build_graph,
        storage: Storage | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.graph_builder = graph_builder
        self.storage = storage or Storage(self.storage_path)
        self.context_manager = ContextManager(self.storage, self.config.context)
        self.context_harness = ContextHarness(
            self.storage,
            self.config.context_harness,
            self.config.output_dir.rstrip("/\\") + "/context",
            self.memory_path,
        )
        self.memory_runtime = MemoryGovernanceRuntime(
            self.memory_path,
            storage=self.storage,
            min_active_confidence=self.config.memory.min_active_confidence,
        )
        self.integration_catalog = IntegrationCatalog.build(self.config)
        self.integration_catalog.register_all()
        self.skill_catalog = self.integration_catalog.skill_catalog
        self.skill_registry = self.integration_catalog.skill_registry
        self.mcp_manager = self.integration_catalog.mcp_manager
        self.mcp_tool_names = self.integration_catalog.mcp_tool_names

    @property
    def storage_path(self) -> str:
        return str(self.config.output_dir.rstrip("/\\") + "/agent.sqlite")

    @property
    def memory_path(self) -> str:
        return str(self.config.output_dir.rstrip("/\\") + "/memory.sqlite")

    def build_context(self, task: str, session_id: str = ""):
        return self.context_harness.prepare(
            ContextRequest(
                session_id=session_id,
                task=task,
                node="planner",
                namespace=session_id,
                include_memory=self.config.memory.enabled,
            )
        )

    def build_harness_context(self, task: str, session_id: str = "", node: str = "planner"):
        return self.context_harness.prepare(
            ContextRequest(session_id=session_id, task=task, node=node, namespace=session_id)
        )

    def run(self, options: RuntimeOptions) -> RunResult:
        if options.team:
            if not options.task:
                raise ValueError("task is required for team mode")
            from src.team import TeamRuntime

            team_result = TeamRuntime(self.config, llm=self.llm, storage=self.storage).run(
                task=options.task,
                team_name=options.team,
                auto_approve=options.auto_approve,
                output_format=options.output_format,
                no_memory=options.no_memory,
            )
            return RunResult(
                run_id=team_result.team_run_id,
                session_id=team_result.session_id,
                status=team_result.status,
                final_answer=team_result.final_answer,
                acceptance={"passed": team_result.status == "completed", "issues": []},
                events=team_result.events,
                tool_calls=[],
                artifacts=[],
                state={"user_task": options.task, "execution_log": [], "team_run_id": team_result.team_run_id},
                team=team_result.__dict__,
            )

        memory_enabled = self.config.memory.enabled and not options.no_memory
        if memory_enabled:
            self.memory_runtime.consolidator.migrate_legacy()
        memory = load_memory(self.memory_path) if memory_enabled else []
        run_id = options.resume or options.run_id or create_run_id()
        start_at = "planner"
        self._full_model = options.full_model

        if options.resume:
            state = load_checkpoint(options.resume, db_path=self.config.checkpoint_path)
            state["memory"] = memory
            state["memory_enabled"] = memory_enabled
            state["session_summary"] = self.storage.get_session_summary(state.get("session_id", ""))
            state["auto_approve"] = options.auto_approve
            start_at = "summarizer" if state.get("is_complete") and not state.get("final_answer") else "executor"
            if state.get("final_answer"):
                return self._result_from_state(run_id, state, "completed")
        else:
            if not options.task:
                raise ValueError("task is required unless resume is used")
            state = self._new_state(options, memory)
            if memory_enabled:
                self.memory_runtime.hot_write_user_task(
                    options.task,
                    namespace=self.config.memory.namespace,
                    session_id=state.get("session_id", ""),
                )

        checkpoint_parent: str | None = None

        def checkpoint_callback(node_name, checkpoint_state):
            nonlocal checkpoint_parent
            checkpoint_parent = save_checkpoint(
                run_id,
                checkpoint_state,
                "running",
                node_name,
                db_path=self.config.checkpoint_path,
                parent_checkpoint_id=checkpoint_parent,
            )

        checkpoint_parent = save_checkpoint(run_id, state, "running", "initial", db_path=self.config.checkpoint_path)
        self._active_session_id = state.get("session_id", "")
        app = self._build_runtime_graph(checkpoint_callback, start_at)
        try:
            result_state = app.invoke(state)
            save_checkpoint(
                run_id,
                result_state,
                "completed",
                "final",
                db_path=self.config.checkpoint_path,
                parent_checkpoint_id=checkpoint_parent,
            )
            structured_summary = self.context_harness.compact(result_state)
            self.context_harness.store.offload(
                __import__("json").dumps(structured_summary.__dict__, ensure_ascii=False, indent=2),
                {
                    "kind": "summary",
                    "summary": f"Structured session summary for {result_state.get('user_task', '')[:120]}",
                    "source_ref": result_state.get("session_id", ""),
                    "scope": result_state.get("session_id", ""),
                    "trust": "verified",
                    "session_id": result_state.get("session_id", ""),
                },
            )
            self._complete_session(result_state, options.output_format)
            if memory_enabled and options.consolidate_memory:
                next_memory = remember_interaction(
                    memory,
                    result_state.get("user_task", options.task or ""),
                    result_state.get("final_answer", ""),
                    result_state.get("tool_results", []),
                )
                save_memory(next_memory, self.memory_path)
                self.memory_runtime.consolidate_run(
                    user_task=result_state.get("user_task", options.task or ""),
                    final_answer=result_state.get("final_answer", ""),
                    tool_results=result_state.get("tool_results", []),
                    acceptance=result_state.get("acceptance", {}),
                    namespace=self.config.memory.namespace,
                    session_id=result_state.get("session_id", ""),
                )
            return self._result_from_state(run_id, result_state, "completed")
        except Exception:
            save_checkpoint(run_id, state, "failed", "error", db_path=self.config.checkpoint_path)
            raise

    def _new_state(self, options: RuntimeOptions, memory: list[dict]) -> dict[str, Any]:
        base_session_id = options.session_id
        if options.continue_session:
            base_session_id = self.storage.latest_session_id()
            if not base_session_id:
                raise ValueError("没有可继续的 session")
        parent_id = base_session_id if options.fork else None
        session_id = new_session_id() if options.fork else base_session_id
        session = create_or_continue_session(
            storage=self.storage,
            task=options.task or "",
            agent=self.config.default_agent,
            output_format=options.output_format,
            session_id=session_id,
            parent_id=parent_id,
        )
        enabled_skills = self.integration_catalog.skill_provider.select_skills(options.task or "")
        permission_rules = [
            {
                "permission": rule.permission,
                "pattern": rule.pattern,
                "action": rule.action,
                "scope": rule.scope,
                "source": rule.source,
            }
            for rule in permissions_for_agent(self.config) + self.integration_catalog.permission_rules(
                self.integration_catalog.skill_provider.tool_names_for_task(options.task or "")
            )
        ]
        enabled_tools = self.integration_catalog.enabled_tool_names(options.task or "", self.config.default_agent)
        context = self.context_harness.prepare(
            ContextRequest(
                session_id=session.session_id,
                task=options.task or "",
                node="planner",
                agent=self.config.default_agent,
                namespace=session.session_id,
                include_memory=self.config.memory.enabled and not options.no_memory,
            )
        )
        return make_initial_state(
            options.task or "",
            memory=memory,
            auto_approve=options.auto_approve,
            session_id=session.session_id,
            message_id=session.message_id,
            session_summary=context.system_context,
            agent=self.config.default_agent,
            permission_rules=permission_rules,
            output_format=options.output_format,
            storage_path=self.storage_path,
            output_dir=self.config.output_dir,
            context_harness_config=self.config.context_harness.model_dump(),
            context_namespace=options.context_namespace or session.session_id,
            context_snapshot_id=context.snapshot_id,
            enabled_skills=enabled_skills,
            enabled_tools=enabled_tools,
            skill_instructions=self.integration_catalog.prompt_for_tools(enabled_tools),
            memory_enabled=self.config.memory.enabled and not options.no_memory,
        )

    def _build_runtime_graph(self, checkpoint_callback, start_at: str):
        llm = ModelUsageRecorder(self.llm, self.storage, getattr(self, "_active_session_id", "")) if self.llm is not None else None
        if llm is not None and getattr(self, "_full_model", False):
            return self.graph_builder(
                llm=llm,
                executor_llm=llm,
                verifier_llm=llm,
                summarizer_llm=llm,
                checkpoint_callback=checkpoint_callback,
                start_at=start_at,
            )
        if llm is not None:
            return self.graph_builder(
                llm=llm,
                executor_llm=llm,
                verifier_llm=None,
                summarizer_llm=None,
                checkpoint_callback=checkpoint_callback,
                start_at=start_at,
            )
        return self.graph_builder(
            llm=None,
            executor_llm=None,
            verifier_llm=None,
            summarizer_llm=None,
            checkpoint_callback=checkpoint_callback,
            start_at=start_at,
        )

    def _complete_session(self, result: dict, output_format: str) -> None:
        session_id = result.get("session_id")
        if not session_id:
            return
        message_id = new_id("msg_assistant")
        self.storage.add_message(message_id, session_id, "assistant", result.get("final_answer", ""))
        self.storage.add_message_part(new_id("part"), message_id, session_id, "text", {"role": "assistant", "content": result.get("final_answer", "")})
        self.storage.update_session_status(session_id, "completed")
        self.storage.compact_session(session_id)
        data = {"final_answer": result.get("final_answer", "")}
        self.storage.add_event(session_id, "session.completed", data)
        if output_format == "json":
            from src.events import Event

            print(Event("session.completed", session_id, data).to_json())

    def _result_from_state(self, run_id: str, state: dict[str, Any], status: str) -> RunResult:
        session_id = state.get("session_id", "")
        tool_calls = self.storage.list_tool_calls(session_id) if session_id else []
        events = self.storage.list_events(session_id) if session_id else []
        artifacts = []
        for item in state.get("tool_results", []):
            if item.get("raw_output_path"):
                artifacts.append(item["raw_output_path"])
            artifacts.extend(item.get("attachments") or [])
        return RunResult(
            run_id=run_id,
            session_id=session_id,
            status=status,
            final_answer=state.get("final_answer", ""),
            acceptance=state.get("acceptance", {}),
            events=events,
            tool_calls=tool_calls,
            artifacts=artifacts,
            state=state,
        )
