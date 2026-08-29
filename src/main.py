import argparse
import json
import os
import os
import sys
from dataclasses import asdict

from src.cli.json_args import parse_json_object
from src.cli.renderers import print_output


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _build_llm(model: str | None):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("缺少 langchain-openai，请先安装 requirements.txt") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("未找到 OPENAI_API_KEY，将使用本地 fallback 逻辑运行。")
        return None

    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def main() -> int:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(description="Workflow Agent CLI")
    parser.add_argument("task", nargs="?", help="要交给 agent 的任务")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 agent_config.json")
    parser.add_argument("--agent", default=None, choices=["build", "plan", "research"], help="Agent 模式")
    parser.add_argument("--team", default=None, choices=["default-dev-team", "hyperplan", "research-build"], help="Team Mode 团队名称")
    parser.add_argument("--model", default=None, help="OpenAI 模型名，例如 gpt-4o-mini")
    parser.add_argument("--offline", action="store_true", help="强制使用本地 fallback 逻辑，不调用模型 API")
    parser.add_argument("--full-model", action="store_true", help="所有节点都允许调用模型；默认 fast 模式只让 planner/executor 调模型")
    parser.add_argument("--yes", action="store_true", help="自动批准需要确认的工具调用")
    parser.add_argument("--format", choices=["default", "json"], default="default", help="输出格式")
    parser.add_argument("--show-memory", action="store_true", help="运行前显示已保存的记忆")
    parser.add_argument("--no-memory", action="store_true", help="本次运行不读取也不保存记忆")
    parser.add_argument("--session", default=None, help="继续指定 session id")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="继续最近的 session")
    parser.add_argument("--fork", action="store_true", help="从指定或最近 session fork 一个新 session")
    parser.add_argument("--list-sessions", action="store_true", help="列出最近 session")
    parser.add_argument("--run-id", default=None, help="指定本次运行的 checkpoint run_id")
    parser.add_argument("--resume", default=None, help="从指定 run_id 的 checkpoint 恢复运行")
    parser.add_argument("--list-runs", action="store_true", help="列出最近的 checkpoint 运行记录")
    parser.add_argument("--inspect-run", default=None, help="查看 run checkpoint 详情")
    parser.add_argument("--inspect-session", default=None, help="查看 session 详情")
    parser.add_argument("--events", default=None, help="查看 session 事件")
    parser.add_argument("--show-context", default=None, help="展示某个任务会使用的上下文")
    parser.add_argument("--list-permissions", action="store_true", help="列出权限决策记录")
    parser.add_argument("--revoke-permission", type=int, default=None, help="撤销指定权限记录")
    parser.add_argument("--team-status", default=None, help="查看 team run 状态")
    parser.add_argument("--team-events", default=None, help="查看 team run 事件")
    parser.add_argument("--team-tasks", default=None, help="查看 team run 任务板")
    parser.add_argument("--team-messages", default=None, help="查看 team run mailbox")
    parser.add_argument("--list-skills", action="store_true", help="列出可用 skills")
    parser.add_argument("--inspect-skill", default=None, help="查看指定 skill 详情")
    parser.add_argument("--validate-skills", action="store_true", help="校验项目内 skill 包")
    parser.add_argument("--run-skill-tool", default=None, help="直接运行动态 skill 工具")
    parser.add_argument("--skill-args", default="{}", help="--run-skill-tool 使用的 JSON 参数")
    parser.add_argument("--list-tool-providers", action="store_true", help="列出工具来源 providers")
    parser.add_argument("--inspect-tool-provider", default=None, help="查看指定工具来源 provider")
    parser.add_argument("--list-tools", action="store_true", help="列出当前工具清单和 provider 元数据")
    parser.add_argument("--doctor-sandbox", action="store_true", help="检查 Docker skill 沙箱状态")
    parser.add_argument("--rag-health", action="store_true", help="检查 RAG 服务或本地知识库状态")
    parser.add_argument("--mcp-health", action="store_true", help="检查 MCP server 状态")
    parser.add_argument("--list-mcp", action="store_true", help="列出 MCP servers")
    parser.add_argument("--inspect-mcp", default=None, help="查看指定 MCP server 配置")
    parser.add_argument("--list-mcp-tools", default=None, help="列出指定 MCP server 工具")
    parser.add_argument("--refresh-mcp", default=None, help="刷新指定 MCP server 工具发现")
    parser.add_argument("--run-mcp-tool", default=None, help="直接运行 MCP 工具，例如 mcp.fetch.fetch")
    parser.add_argument("--mcp-args", default=None, help="--run-mcp-tool 使用的 JSON 参数")
    parser.add_argument("--list-checkpoints", default=None, help="List checkpoint lineage for a run")
    parser.add_argument("--fork-run", default=None, help="Run ID that owns --fork-checkpoint")
    parser.add_argument("--fork-checkpoint", default=None, help="Create a new branch from a checkpoint")
    parser.add_argument("--inspect-context", default=None, help="Inspect a context snapshot")
    parser.add_argument("--context-artifact", default=None, help="Recall a context artifact by id")
    parser.add_argument("--context-stats", default=None, help="Show context feedback and cache events for a session")
    parser.add_argument("--list-memory", action="store_true", help="List governed long-term memory")
    parser.add_argument("--inspect-memory", default=None, help="Inspect a governed memory record")
    parser.add_argument("--memory-candidates", action="store_true", help="List memory candidates awaiting promotion")
    parser.add_argument("--consolidate-memory", action="store_true", help="Run idempotent legacy memory migration")
    parser.add_argument("--invalidate-memory", default=None, help="Invalidate a governed memory record")
    parser.add_argument("--explain-memory-retrieval", default=None, help="Explain memory retrieval for a task")
    parser.add_argument("--memory-stats", action="store_true", help="Show governed memory statistics")
    parser.add_argument("--eval-list", action="store_true", help="List available eval suites")
    parser.add_argument("--eval-run", default=None, help="Run an eval suite")
    parser.add_argument("--eval-report", default=None, help="Inspect an eval report by id")
    parser.add_argument("--eval-compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="Compare two eval reports")
    parser.add_argument("--eval-judge", action="store_true", help="Enable optional LLM judge scoring for eval runs")
    parser.add_argument("--eval-baseline", default=None, help="Compare an eval run against a baseline report")
    parser.add_argument("--eval-threshold", type=float, default=None, help="Override weighted eval pass threshold")
    parser.add_argument("--eval-fail-on-regression", action="store_true", help="Fail eval run when it regresses from baseline")
    parser.add_argument("--eval-export", default=None, help="Export an eval report for an external platform")
    parser.add_argument(
        "--eval-export-platform",
        default="promptfoo",
        choices=["promptfoo", "langsmith", "langfuse", "openai"],
        help="External eval platform export format",
    )
    parser.add_argument("--eval-export-path", default=None, help="Optional export output path")
    parser.add_argument("--benchmark-list", action="store_true", help="List available benchmark suites")
    parser.add_argument("--benchmark-import", choices=["agentbench", "claw"], default=None, help="Import external benchmark tasks")
    parser.add_argument("--benchmark-path", default=None, help="Path for --benchmark-import")
    parser.add_argument("--benchmark-run", default=None, help="Run a benchmark suite")
    parser.add_argument("--benchmark-report", default=None, help="Inspect a benchmark report by id")
    parser.add_argument("--benchmark-doctor", choices=["agentbench", "claw", "deepeval"], default=None, help="Check official benchmark protocol environment")
    parser.add_argument("--benchmark-run-official", choices=["agentbench", "claw"], default=None, help="Run an official-protocol benchmark wrapper")
    parser.add_argument("--benchmark-export-official", default=None, help="Export an official-protocol benchmark report")
    parser.add_argument("--deepeval-mode", choices=["local", "native"], default=None, help="DeepEval scoring mode for eval/benchmark runs")
    parser.add_argument("--claw-workflow-run", default=None, help="Run a Claw-Eval task through this project's workflow runtime")
    parser.add_argument("--claw-workflow-batch", default=None, help="Run a Claw-Eval task suite through this project's workflow runtime")
    parser.add_argument("--claw-mode", choices=["single", "team"], default="single", help="Workflow runtime mode for Claw-Eval adapter")
    parser.add_argument("--claw-config", default="config_workflow_agent.yaml", help="Claw-Eval config path, relative to the Claw checkout by default")
    parser.add_argument("--claw-trace-dir", default="outputs/benchmarks/claw_workflow", help="Output directory for workflow-agent Claw traces")
    parser.add_argument("--split", default="dev", help="Official benchmark split")
    parser.add_argument("--suite", default="", help="Official benchmark suite/environment")
    parser.add_argument("--limit", type=int, default=None, help="Limit official benchmark cases")
    parser.add_argument("--trials", type=int, default=None, help="Official benchmark trial count")
    parser.add_argument("--serve-a2a-role", choices=["researcher", "builder", "reviewer"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--a2a-agents", action="store_true")
    parser.add_argument("--doctor-observability", action="store_true")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        from src.checkpoint import fork_checkpoint, list_checkpoints, list_runs, load_checkpoint
        from src.config import load_config
        from src.memory import load_memory
        from src.runtime import AgentRuntime, RuntimeOptions
        from src.storage import Storage

        load_dotenv()
        app_config = load_config(args.config, args.agent)
        if args.serve_a2a_role:
            from src.a2a_service import serve_role
            serve_role(app_config, args.serve_a2a_role, args.host, args.port)
            return 0
        if args.deepeval_mode:
            app_config.evals.deepeval_mode = args.deepeval_mode
        llm = None if args.offline else _build_llm(_resolve_model(args.model, app_config.model))
        storage = Storage(_storage_path(app_config))
        runtime = AgentRuntime(app_config, llm=llm, storage=storage)
    except ImportError as exc:
        print("依赖未安装。请先运行：pip install -r requirements.txt")
        print(f"错误详情：{exc}")
        return 1

    if args.a2a_agents:
        items = []
        for role, agent in app_config.a2a.agents.items():
            items.append({"role": role, "enabled": agent.enabled, "agent_card_url": agent.agent_card_url, "workspace_root": agent.workspace_root})
        _print(args.format, json.dumps({"items": items}, ensure_ascii=False), {"items": items})
        return 0

    if args.doctor_observability:
        payload = {"provider": type(runtime.trace_provider).__name__, "jsonl_fallback": app_config.observability.jsonl_fallback, "otlp_configured": bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))}
        _print(args.format, json.dumps(payload, ensure_ascii=False), payload)
        return 0

    if args.list_checkpoints:
        checkpoints = list_checkpoints(args.list_checkpoints, db_path=app_config.checkpoint_path)
        text = "\n".join(
            f"{item['checkpoint_id']} | parent={item['parent_checkpoint_id']} | branch={item['branch_id']} | node={item['node']}"
            for item in checkpoints
        ) or "No checkpoints."
        _print(args.format, text, {"type": "checkpoint.lineage", "run_id": args.list_checkpoints, "items": checkpoints})
        return 0

    if args.fork_checkpoint:
        if not args.fork_run:
            parser.error("--fork-checkpoint requires --fork-run")
        branch_id, state = fork_checkpoint(args.fork_run, args.fork_checkpoint, db_path=app_config.checkpoint_path)
        _print(args.format, f"Created branch {branch_id} from {args.fork_checkpoint}.", {"type": "checkpoint.forked", "branch_id": branch_id, "state": state})
        return 0

    if args.inspect_context:
        snapshot = storage.get_context_snapshot(args.inspect_context)
        text = json.dumps(snapshot, ensure_ascii=False, indent=2) if snapshot else f"Unknown context snapshot: {args.inspect_context}"
        _print(args.format, text, {"type": "context.snapshot", "snapshot": snapshot})
        return 0

    if args.context_artifact:
        from src.context_harness import ContextStore

        try:
            content = ContextStore(storage, app_config.output_dir.rstrip("/\\") + "/context").recall(args.context_artifact)
            artifact = storage.get_context_artifact(args.context_artifact)
            _print(args.format, content, {"type": "context.artifact", "artifact": artifact, "content": content})
        except Exception as exc:
            _print(args.format, f"Unable to recall context artifact: {exc}", {"type": "context.artifact.error", "artifact_id": args.context_artifact, "error": str(exc)})
        return 0

    if args.context_stats:
        events = [
            item for item in storage.list_events(args.context_stats)
            if item["type"].startswith("context.") or item["type"].startswith("model.cache")
        ]
        feedback = storage.list_context_feedback(args.context_stats)
        text = "\n".join(f"{item['created_at']} | {item['type']} | {item['data']}" for item in events) or "No context events."
        _print(args.format, text, {"type": "context.stats", "session_id": args.context_stats, "events": events, "feedback": feedback})
        return 0

    if args.list_memory:
        _print_governed_memory(runtime.memory_runtime.store.list(), args.format, "memory.list")
        return 0

    if args.memory_candidates:
        _print_governed_memory(runtime.memory_runtime.store.list(status="candidate"), args.format, "memory.candidates")
        return 0

    if args.inspect_memory:
        record = runtime.memory_runtime.store.get(args.inspect_memory)
        payload = record.__dict__ if record else None
        _print(args.format, json.dumps(payload, ensure_ascii=False, indent=2) if payload else f"Unknown memory: {args.inspect_memory}", {"type": "memory.inspect", "record": payload})
        return 0

    if args.consolidate_memory:
        migrated = runtime.memory_runtime.consolidator.migrate_legacy()
        _print(args.format, f"Legacy memory migration completed. Migrated interactions: {migrated}", {"type": "memory.consolidated", "migrated": migrated})
        return 0

    if args.invalidate_memory:
        updated = runtime.memory_runtime.invalidate(args.invalidate_memory)
        _print(args.format, "Memory invalidated." if updated else f"Unknown memory: {args.invalidate_memory}", {"type": "memory.invalidated", "memory_id": args.invalidate_memory, "updated": updated})
        return 0

    if args.explain_memory_retrieval:
        items = runtime.memory_runtime.explain(args.explain_memory_retrieval, node="planner", namespace=app_config.memory.namespace)
        text = "\n".join(
            f"{item['record']['id']} | score={item['score']:.3f} | {item['record']['memory_type']} | {item['record']['subject']} | {item['reasons']}"
            for item in items
        ) or "No active memory matched."
        _print(args.format, text, {"type": "memory.retrieval.explain", "query": args.explain_memory_retrieval, "items": items})
        return 0

    if args.memory_stats:
        stats = runtime.memory_runtime.store.stats()
        usage = storage.list_memory_usage()
        stats["usage"] = {
            "total": len(usage),
            "selected": sum(1 for item in usage if item.get("selected")),
            "cited_in_answer": sum(1 for item in usage if item.get("cited_in_answer")),
            "helped_acceptance": sum(1 for item in usage if item.get("helped_acceptance")),
            "feedback": sum(1 for item in usage if item.get("feedback")),
        }
        _print(args.format, json.dumps(stats, ensure_ascii=False, indent=2), {"type": "memory.stats", **stats})
        return 0

    if args.eval_list:
        from src.evals import EvalHarness

        harness = EvalHarness(app_config, llm=llm)
        suites = harness.list_suites()
        text = "\n".join(f"{item['suite']} | cases={item['cases']}" for item in suites) or "No eval suites."
        _print(args.format, text, {"type": "eval.suites", "suites": suites})
        return 0

    if args.eval_run:
        from src.evals import EvalHarness

        harness = EvalHarness(app_config, llm=llm, judge_enabled=args.eval_judge)
        try:
            report = harness.run_suite(
                args.eval_run,
                offline=args.offline,
                output_format=args.format,
                threshold=args.eval_threshold,
                baseline_run_id=args.eval_baseline,
                fail_on_regression=args.eval_fail_on_regression,
            )
        except ValueError as exc:
            parser.error(str(exc))
        text = _format_eval_report(report.__dict__)
        _print(args.format, text, {"type": "eval.report", **report.__dict__})
        return 0

    if args.eval_report:
        from src.evals import EvalHarness

        harness = EvalHarness(app_config, llm=llm)
        report = harness.load_report(args.eval_report)
        text = _format_eval_report(report) if report else f"Unknown eval report: {args.eval_report}"
        _print(args.format, text, {"type": "eval.report.inspect", "report": report})
        return 0

    if args.eval_compare:
        from src.evals import EvalHarness

        harness = EvalHarness(app_config, llm=llm)
        try:
            comparison = harness.compare_reports(args.eval_compare[0], args.eval_compare[1])
        except ValueError as exc:
            parser.error(str(exc))
        _print(args.format, json.dumps(comparison, ensure_ascii=False, indent=2), {"type": "eval.compare", **comparison})
        return 0

    if args.eval_export:
        from src.evals import EvalHarness

        harness = EvalHarness(app_config, llm=llm)
        try:
            exported = harness.export_report(args.eval_export, args.eval_export_platform, args.eval_export_path)
        except ValueError as exc:
            parser.error(str(exc))
        text = "\n".join(
            [
                f"Exported: {exported['path']}",
                f"Platform: {exported['platform']}",
                f"Items: {exported['items']}",
                exported.get("note", ""),
            ]
        ).strip()
        _print(args.format, text, {"type": "eval.export", **exported})
        return 0

    if args.benchmark_list:
        from src.benchmarks import BenchmarkHarness

        harness = BenchmarkHarness(app_config, llm=llm)
        suites = harness.list_suites()
        text = "\n".join(f"{item['suite']} | cases={item['cases']} | sources={','.join(item['sources'])}" for item in suites) or "No benchmark suites."
        _print(args.format, text, {"type": "benchmark.suites", "suites": suites})
        return 0

    if args.benchmark_doctor:
        from src.benchmark_protocols import make_protocol_runner

        result = make_protocol_runner(args.benchmark_doctor, app_config).doctor()
        text = _format_protocol_doctor(result.__dict__)
        _print(args.format, text, {"type": "benchmark.protocol.doctor", **result.__dict__})
        return 0

    if args.benchmark_import:
        from src.benchmarks import BenchmarkHarness

        if not args.benchmark_path:
            parser.error("--benchmark-import requires --benchmark-path")
        harness = BenchmarkHarness(app_config, llm=llm)
        try:
            imported = harness.import_cases(args.benchmark_import, args.benchmark_path)
        except ValueError as exc:
            parser.error(str(exc))
        text = f"Imported {imported['imported']} cases to {imported['path']} (unsupported={imported['unsupported']})."
        _print(args.format, text, {"type": "benchmark.imported", **imported})
        return 0

    if args.benchmark_run:
        from src.benchmarks import BenchmarkHarness

        harness = BenchmarkHarness(app_config, llm=llm)
        try:
            report = harness.run_suite(args.benchmark_run, offline=args.offline, output_format=args.format)
        except ValueError as exc:
            parser.error(str(exc))
        payload = report.__dict__
        text = _format_benchmark_report(payload)
        _print(args.format, text, {"type": "benchmark.report", **payload})
        return 0

    if args.benchmark_run_official:
        from src.benchmark_protocols import make_protocol_runner

        runner = make_protocol_runner(args.benchmark_run_official, app_config)
        report = runner.run(
            split=args.split,
            suite=args.suite or ("general" if args.benchmark_run_official == "claw" else "dbbench"),
            limit=args.limit,
            trials=args.trials,
        )
        payload = report.__dict__
        text = _format_official_benchmark_report(payload)
        _print(args.format, text, {"type": "benchmark.official.report", **payload})
        return 0

    if args.claw_workflow_run or args.claw_workflow_batch:
        from src.claw_workflow import ClawWorkflowRunner

        runner = ClawWorkflowRunner(app_config, llm=llm, storage=storage)
        if args.claw_workflow_batch:
            report = runner.run_batch(
                args.claw_workflow_batch,
                claw_config=args.claw_config,
                trace_dir=args.claw_trace_dir,
                trials=args.trials or 3,
                grade=True,
                auto_approve=args.yes,
                mode=args.claw_mode,
                team_name=args.team or "default-dev-team",
            )
            payload = {
                "type": "claw.workflow.report",
                "suite": report.suite,
                "mode": report.mode,
                "trials": report.trials,
                "tasks": [item.__dict__ for item in report.tasks],
                "results": [item.__dict__ for item in report.results],
            }
        else:
            results = runner.run(
                args.claw_workflow_run,
                claw_config=args.claw_config,
                trace_dir=args.claw_trace_dir,
                trials=args.trials or 1,
                grade=True,
                auto_approve=args.yes,
                mode=args.claw_mode,
                team_name=args.team or "default-dev-team",
            )
            payload = {"type": "claw.workflow.report", "mode": args.claw_mode, "results": [item.__dict__ for item in results]}
        text = _format_claw_workflow_report(payload)
        _print(args.format, text, payload)
        return 0

    if args.benchmark_report:
        from src.benchmarks import BenchmarkHarness

        harness = BenchmarkHarness(app_config, llm=llm)
        report = harness.load_report(args.benchmark_report)
        text = _format_benchmark_report(report) if report else f"Unknown benchmark report: {args.benchmark_report}"
        _print(args.format, text, {"type": "benchmark.report.inspect", "report": report})
        return 0

    if args.benchmark_export_official:
        from src.benchmark_protocols import make_protocol_runner

        exported = None
        for protocol in ["agentbench", "claw", "deepeval"]:
            exported = make_protocol_runner(protocol, app_config).export_report(args.benchmark_export_official)
            if exported:
                break
        text = f"Exported: {exported['path']}" if exported else f"Unknown official benchmark report: {args.benchmark_export_official}"
        _print(args.format, text, {"type": "benchmark.official.export", "exported": exported})
        return 0

    if args.list_sessions:
        _print_sessions(storage, args.format)
        return 0

    if args.list_runs:
        _print_runs(app_config, args.format, list_runs)
        return 0

    if args.inspect_run:
        state = load_checkpoint(args.inspect_run, db_path=app_config.checkpoint_path)
        _print(args.format, _format_inspect_run(args.inspect_run, state), {"type": "inspect.run", "run_id": args.inspect_run, "state": state})
        return 0

    if args.inspect_session:
        _print(args.format, _format_inspect_session(storage, args.inspect_session), {"type": "inspect.session", "session_id": args.inspect_session, "messages": storage.list_messages(args.inspect_session), "parts": storage.list_message_parts(args.inspect_session), "tool_calls": storage.list_tool_calls(args.inspect_session)})
        return 0

    if args.events:
        events = storage.list_events(args.events)
        _print(args.format, "\n".join(f"{item['created_at']} | {item['type']} | {item['data']}" for item in events), {"type": "events", "session_id": args.events, "events": events})
        return 0

    if args.show_context:
        bundle = runtime.build_context(args.show_context, args.session or "")
        payload = {"type": "context", **asdict(bundle)}
        skills_text = "\n".join(f"- {name} ({skill.source}): {skill.description}" for name, skill in runtime.skill_registry.items())
        selected_tools = runtime.integration_catalog.enabled_tool_names(args.show_context, runtime.config.default_agent)
        providers_text = "\n".join(f"- {item.name}: {item.kind}" for item in runtime.integration_catalog.provider_health())
        tools_text = "\n".join(f"- {name}" for name in selected_tools)
        manifest_text = _format_context_manifest(getattr(bundle, "manifest", {}) or {})
        _print(
            args.format,
            bundle.system_context
            + f"\n\nContext Manifest:\n{manifest_text}"
            + f"\n\nSelected Skills:\n{skills_text}"
            + f"\n\nSelected Tool Providers:\n{providers_text}"
            + f"\n\nSelected Tools:\n{tools_text}"
            + f"\n\n估算字符数：{bundle.estimated_chars}",
            {**payload, "skills": {name: _skill_payload(skill) for name, skill in runtime.skill_registry.items()}, "tool_providers": [item.__dict__ for item in runtime.integration_catalog.provider_health()], "selected_tools": selected_tools},
        )
        return 0

    if args.list_skills:
        _print_skills(runtime, args.format)
        return 0

    if args.inspect_skill:
        _print_skill(runtime, args.inspect_skill, args.format)
        return 0

    if args.validate_skills:
        _print_skill_validation(runtime, args.format)
        return 0

    if args.run_skill_tool:
        _run_skill_tool(runtime, args.run_skill_tool, args.skill_args, args.yes, args.format)
        return 0

    if args.list_tool_providers:
        _print_tool_providers(runtime, args.format)
        return 0

    if args.inspect_tool_provider:
        _print_tool_provider(runtime, args.inspect_tool_provider, args.format)
        return 0

    if args.list_tools:
        _print_tools(runtime, args.task or "", args.format)
        return 0

    if args.doctor_sandbox:
        _print_sandbox_doctor(runtime, args.format)
        return 0

    if args.rag_health:
        _print_rag_health(args.format)
        return 0

    if args.mcp_health:
        _print_mcp_health(runtime, args.format)
        return 0

    if args.list_mcp:
        _print_mcp_list(runtime, args.format)
        return 0

    if args.inspect_mcp:
        _print_mcp_inspect(runtime, args.inspect_mcp, args.format)
        return 0

    if args.list_mcp_tools:
        _print_mcp_tools(runtime, args.list_mcp_tools, refresh=False, output_format=args.format)
        return 0

    if args.refresh_mcp:
        _print_mcp_tools(runtime, args.refresh_mcp, refresh=True, output_format=args.format)
        return 0

    if args.run_mcp_tool:
        _run_mcp_tool(runtime, args.run_mcp_tool, args.mcp_args or args.skill_args, args.yes, args.format)
        return 0

    if args.list_permissions:
        permissions = storage.list_permissions()
        text = "\n".join(f"{item['id']} | {item['action']} | {item['permission']}:{item['pattern']} | {item['scope']} | {item['source']}" for item in permissions) or "暂无权限记录。"
        _print(args.format, text, {"type": "permissions", "items": permissions})
        return 0

    if args.revoke_permission is not None:
        revoked = storage.revoke_permission(args.revoke_permission)
        _print(args.format, "已撤销权限记录。" if revoked else "未找到权限记录。", {"type": "permission.revoked", "id": args.revoke_permission, "revoked": revoked})
        return 0

    if args.team_status:
        _print_team_status(storage, args.team_status, args.format)
        return 0

    if args.team_events:
        run = storage.get_team_run(args.team_events)
        events = storage.list_events(run["session_id"]) if run else []
        _print(args.format, "\n".join(f"{item['created_at']} | {item['type']} | {item['data']}" for item in events), {"type": "team.events", "team_run_id": args.team_events, "events": events})
        return 0

    if args.team_tasks:
        tasks = storage.list_team_tasks(args.team_tasks)
        text = "\n".join(
            (
                f"{item['id']} | {item['status']} | {item['assigned_to']} | {item['title']} "
                f"| depends_on={item.get('metadata', {}).get('depends_on', [])} "
                f"| strategy={item.get('metadata', {}).get('assignment_strategy', 'unknown')} "
                f"| node_type={item.get('metadata', {}).get('node_type', 'task')} "
                f"| condition={item.get('metadata', {}).get('condition', 'always')} "
                f"| optional={item.get('metadata', {}).get('optional', False)} "
                f"| replan_round={item.get('metadata', {}).get('replan_round', 0)} "
                f"| skip_reason={item.get('metadata', {}).get('skip_reason', '')} "
                f"| reason={item.get('metadata', {}).get('assignment_reason', '')}"
            )
            for item in tasks
        ) or "No team tasks."
        _print(args.format, text, {"type": "team.tasks", "team_run_id": args.team_tasks, "tasks": tasks})
        return 0

    if args.team_messages:
        messages = storage.list_team_messages(args.team_messages)
        text = "\n".join(f"{item['created_at']} | {item['sender']} -> {item['recipient']} | {item['payload']}" for item in messages) or "No team messages."
        _print(args.format, text, {"type": "team.messages", "team_run_id": args.team_messages, "messages": messages})
        return 0

    if args.show_memory:
        memory = [] if args.no_memory else load_memory(runtime.memory_path)
        _print_memory(memory, args.format)

    try:
        result = runtime.run(
            RuntimeOptions(
                task=args.task,
                agent=args.agent,
                run_id=args.run_id,
                resume=args.resume,
                session_id=args.session,
                continue_session=args.continue_session,
                fork=args.fork,
                auto_approve=args.yes,
                output_format=args.format,
                no_memory=args.no_memory,
                full_model=args.full_model,
                team=args.team,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    except Exception as exc:
        if llm is None:
            raise
        print(f"模型调用失败，将自动切换到本地 fallback 逻辑。错误：{exc}")
        runtime = AgentRuntime(app_config, llm=None, storage=storage)
        result = runtime.run(
            RuntimeOptions(
                task=args.task,
                agent=args.agent,
                run_id=args.run_id,
                resume=args.resume,
                session_id=args.session,
                continue_session=args.continue_session,
                fork=args.fork,
                auto_approve=args.yes,
                output_format=args.format,
                no_memory=args.no_memory,
                full_model=False,
                team=args.team,
            )
        )

    if args.format == "json":
        print(json.dumps({"type": "run.result", **result.__dict__}, ensure_ascii=False))
        return 0

    state = result.state
    print("\n=== Run ID ===")
    print(result.run_id)
    print("\n=== Session ID ===")
    print(result.session_id)
    if result.team:
        print("\n=== Team Run ID ===")
        print(result.team.get("team_run_id"))
        print("\n=== Team Members ===")
        for item in result.team.get("members", []):
            print(f"{item['name']} | {item['role']} | {item['status']} | {item['session_id']}")
        print("\n=== Team Tasks ===")
        for item in result.team.get("tasks", []):
            print(f"{item['status']} | {item['assigned_to']} | {item['title']}")
    print("\n=== 模型模式 ===")
    print("full-model" if args.full_model else "fast")
    print("\n=== 执行日志 ===")
    for item in state.get("execution_log", []):
        print(item)
    print("\n=== 验收 ===")
    print(result.acceptance or {"passed": True, "issues": []})
    print("\n=== 最终结果 ===")
    print(result.final_answer or "没有生成最终结果。")
    print("\n=== 记忆 ===")
    print("本次运行已禁用记忆读写。" if args.no_memory else "已更新本次任务记忆。")
    return 0


def _print(output_format: str, text: str, payload: dict) -> None:
    print_output(output_format, text, payload)


def _format_eval_report(report: dict | None) -> str:
    if not report:
        return "No eval report."
    summary = report.get("summary", {})
    lines = [
        f"Eval Run ID: {report.get('eval_run_id')}",
        f"Suite: {report.get('suite')}",
        f"Passed: {report.get('passed')}",
        f"Cases: {summary.get('passed', 0)}/{summary.get('total', 0)}",
        f"Hard Passed: {summary.get('hard_passed', summary.get('passed', 0))}/{summary.get('total', 0)}",
        f"Average Quality: {summary.get('average_quality_score', '-')}",
        f"Issues: {summary.get('issue_counts', {})}",
        f"Duration: {report.get('duration_ms')} ms",
    ]
    services = (report.get("environment") or {}).get("services") or {}
    if services:
        lines.append("")
        lines.append("Environment:")
        for name, status in services.items():
            lines.append(f"- {name}: {'ok' if status.get('ok') else 'failed'} | {status.get('code') or status.get('error') or ''}")
    failed = [
        item for item in report.get("case_results", [])
        if not item.get("score", {}).get("passed")
    ]
    if failed:
        lines.append("")
        lines.append("Failed Cases:")
        for item in failed:
            score = item.get("score", {})
            lines.append(f"- {item.get('case', {}).get('id')}: {'; '.join(score.get('issues', []))}")
    return "\n".join(lines)


def _format_benchmark_report(report: dict | None) -> str:
    if not report:
        return "No benchmark report."
    lines = [
        f"Benchmark Run ID: {report.get('benchmark_run_id')}",
        f"Suite: {report.get('suite')}",
        f"Source: {report.get('source')}",
        f"Success Rate: {report.get('success_rate')}",
        f"Duration: {report.get('duration_ms')} ms",
        "",
        "Metric Summary:",
    ]
    metrics = report.get("metric_summary", {})
    lines.extend(f"- {key}: {value}" for key, value in metrics.items())
    lines.append("")
    lines.append("Failure Breakdown:")
    failures = report.get("failure_breakdown", {})
    if failures:
        lines.extend(f"- {key}: {value}" for key, value in failures.items())
    else:
        lines.append("- none")
    return "\n".join(lines)


def _format_protocol_doctor(result: dict) -> str:
    lines = [
        f"Protocol: {result.get('protocol')}",
        f"OK: {result.get('ok')}",
        "",
        "Checks:",
    ]
    for name, status in (result.get("checks") or {}).items():
        lines.append(f"- {name}: {'ok' if status.get('ok') else 'failed'} | {status.get('error') or status.get('value') or status.get('output') or ''}")
    recommendations = result.get("recommendations") or []
    if recommendations:
        lines.append("")
        lines.append("Recommendations:")
        lines.extend(f"- {item}" for item in recommendations)
    return "\n".join(lines)


def _format_official_benchmark_report(report: dict | None) -> str:
    if not report:
        return "No official benchmark report."
    lines = [
        f"Official Benchmark Run ID: {report.get('benchmark_run_id')}",
        f"Protocol: {report.get('protocol')}",
        f"Status: {report.get('status')}",
        f"Split/Suite: {report.get('split')} / {report.get('suite')}",
        f"Official Score: {report.get('official_score')}",
        f"Local Score: {report.get('local_score')}",
        f"Unsupported Count: {report.get('unsupported_count')}",
        f"Duration: {report.get('duration_ms')} ms",
        "",
        "Environment:",
    ]
    environment = report.get("environment_status") or {}
    checks = environment.get("checks") or {}
    if checks:
        lines.extend(f"- {name}: {'ok' if status.get('ok') else 'failed'}" for name, status in checks.items())
    else:
        lines.append("- none")
    trials = report.get("trial_results") or []
    if trials:
        lines.append("")
        lines.append("Trials:")
        for item in trials:
            lines.append(f"- trial {item.get('trial')}: {item.get('status')} | passed={item.get('passed')} | score={item.get('score')}")
    notes = report.get("notes") or []
    if notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"- {item}" for item in notes)
    return "\n".join(lines)


def _format_claw_workflow_report(report: dict | None) -> str:
    if not report:
        return "No Claw workflow report."
    lines = ["Workflow-Agent Claw-Eval Report"]
    if report.get("suite"):
        lines.extend(
            [
                f"Suite: {report.get('suite')}",
                f"Mode: {report.get('mode')}",
                f"Trials per task: {report.get('trials')}",
                "",
                "Task summary:",
            ]
        )
        for item in report.get("tasks", []):
            lines.append(
                f"- {item.get('task_id')} | pass_rate={item.get('pass_rate'):.2f} "
                f"| native={item.get('native_pass_rate', 0.0):.2f} | repaired={item.get('repaired_pass_rate', 0.0):.2f} "
                f"| mean={item.get('mean_score'):.2f} | best={item.get('best_score'):.2f} "
                f"| trials={item.get('trials')}"
            )
    for item in report.get("results", []):
        lines.extend(
            [
                "",
                f"Task: {item.get('task_id')} trial={item.get('trial')} mode={item.get('mode')}",
                f"Status: {item.get('status')}",
                f"Score: {item.get('task_score')} passed={item.get('passed')} scores={item.get('scores')}",
                f"Native tool success: {item.get('native_tool_success')} adapter_repaired={item.get('adapter_repaired')}",
                f"Trace: {item.get('trace_path')}",
                f"Grade return code: {item.get('grading_returncode')}",
                "Tool calls:",
            ]
        )
        for call in item.get("tool_calls", []):
            lines.append(f"- {call.get('tool')} {call.get('input')} status={call.get('status')}")
        if item.get("error"):
            lines.append(f"Error: {item.get('error')}")
        if item.get("grading_output"):
            lines.append("")
            lines.append("Grading output:")
            lines.append(str(item.get("grading_output")).strip())
        if item.get("final_answer"):
            lines.append("")
            lines.append("Final answer:")
            lines.append(str(item.get("final_answer")).strip())
    return "\n".join(lines)


def _format_context_manifest(manifest: dict) -> str:
    if not manifest:
        return "No context manifest."
    budget = manifest.get("budget") or {}
    versions = manifest.get("versions") or {}
    trust_summary = manifest.get("trust_summary") or {}
    lines = [
        f"Budget: {budget.get('used_tokens', manifest.get('used_tokens', 0))}/{budget.get('budget_tokens', manifest.get('budget_tokens', 0))} tokens",
        f"Versions: {json.dumps(versions, ensure_ascii=False)}",
        f"Trust: {json.dumps(trust_summary, ensure_ascii=False)}",
        "",
        "Selected Memory:",
    ]
    selected_memory = manifest.get("selected_memory") or manifest.get("memory") or []
    if selected_memory:
        for item in selected_memory:
            reason = item.get("reason")
            if isinstance(reason, list):
                reason = "; ".join(str(part) for part in reason)
            lines.append(
                f"- {item.get('memory_id')} | {item.get('type')} | trust={item.get('trust')} "
                f"| score={item.get('score')} | {reason or ''}"
            )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Selected Artifacts:")
    selected_artifacts = manifest.get("selected_artifacts") or manifest.get("artifacts") or []
    if selected_artifacts:
        for item in selected_artifacts:
            lines.append(
                f"- {item.get('artifact_id')} | {item.get('kind')} | recallable={item.get('recallable')} "
                f"| {item.get('summary') or ''}"
            )
    else:
        lines.append("- none")
    dropped = manifest.get("dropped_by_budget") or manifest.get("dropped") or []
    lines.append("")
    lines.append("Dropped By Budget:")
    if dropped:
        for item in dropped:
            lines.append(f"- {item.get('id')} | {item.get('reason')} | tokens={item.get('tokens')}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _print_sessions(storage, output_format: str) -> None:
    sessions = storage.list_sessions()
    if not sessions:
        _print(output_format, "暂无 session。", {"type": "session.empty"})
        return
    text = "\n".join(f"{item['id']} | {item['agent']} | {item['status']} | {item['updated_at']} | {item['title']}" for item in sessions)
    _print(output_format, text, {"type": "session.list", "items": sessions})


def _print_runs(app_config, output_format: str, list_runs_func) -> None:
    runs = list_runs_func(db_path=app_config.checkpoint_path)
    if not runs:
        _print(output_format, "暂无 checkpoint。", {"type": "checkpoint.empty"})
        return
    text = "\n".join(f"{item['run_id']} | {item['status']} | {item['updated_at']} | {item['user_task']}" for item in runs)
    _print(output_format, text, {"type": "checkpoint.list", "items": runs})


def _print_memory(memory: list[dict], output_format: str) -> None:
    if not memory:
        _print(output_format, "暂无记忆。", {"type": "memory.empty"})
        return
    text = "\n".join(f"{index}. {item.get('user_task')} | {(item.get('summary') or item.get('final_answer') or '')[:120]}" for index, item in enumerate(memory, start=1))
    _print(output_format, text, {"type": "memory.list", "items": memory})


def _print_governed_memory(records, output_format: str, event_type: str) -> None:
    items = [record.__dict__ for record in records]
    text = "\n".join(
        f"{item['id']} | {item['status']} | {item['memory_type']} | {item['trust']} | {item['subject']} | {item['content'][:100]}"
        for item in items
    ) or "No governed memory records."
    _print(output_format, text, {"type": event_type, "items": items})


def _print_skills(runtime, output_format: str) -> None:
    items = []
    for name, skill in sorted(runtime.skill_registry.items()):
        items.append(
            {
                "name": name,
                "source": skill.source,
                "description": skill.description,
                "tools": skill.tools,
                "triggers": skill.triggers,
                "path": skill.path,
            }
        )
    text = "\n".join(f"{item['name']} | {item['source']} | tools={','.join(item['tools'])} | {item['description']}" for item in items) or "No skills."
    _print(output_format, text, {"type": "skills.list", "items": items, "errors": [error.__dict__ for error in runtime.skill_catalog.errors]})


def _skill_payload(skill) -> dict:
    return {
        "name": skill.name,
        "source": skill.source,
        "description": skill.description,
        "tools": skill.tools,
        "permissions": [rule.model_dump() for rule in skill.permissions],
        "triggers": skill.triggers,
        "path": skill.path,
        "metadata": skill.metadata,
        "system_prompt": skill.system_prompt,
    }


def _print_skill(runtime, name: str, output_format: str) -> None:
    skill = runtime.skill_registry.get(name)
    if not skill:
        _print(output_format, f"Unknown skill: {name}", {"type": "skill.inspect", "found": False, "name": name})
        return
    payload = {"type": "skill.inspect", "found": True, "skill": _skill_payload(skill)}
    lines = [
        f"Skill: {skill.name}",
        f"Source: {skill.source}",
        f"Description: {skill.description}",
        f"Path: {skill.path or '-'}",
        f"Tools: {', '.join(skill.tools) or '-'}",
        f"Triggers: {', '.join(skill.triggers) or '-'}",
        f"Metadata: {skill.metadata}",
        "",
        "Instructions:",
        skill.system_prompt,
    ]
    _print(output_format, "\n".join(lines), payload)


def _print_skill_validation(runtime, output_format: str) -> None:
    errors = [error.__dict__ for error in runtime.skill_catalog.errors]
    if errors:
        text = "\n".join(f"{item['path']} | {item['message']}" for item in errors)
    else:
        text = f"Skills valid. Loaded project skills: {len(runtime.skill_catalog.packages)}"
    _print(output_format, text, {"type": "skills.validation", "ok": not errors, "errors": errors, "loaded": list(runtime.skill_catalog.packages)})


def _print_tool_providers(runtime, output_format: str) -> None:
    items = [item.__dict__ for item in runtime.integration_catalog.provider_health()]
    text = "\n".join(
        f"{item['name']} | {item['kind']} | {'ok' if item['ok'] else 'failed'} | tools={len(item.get('tools') or [])} | {item.get('error') or ''}"
        for item in items
    ) or "No tool providers."
    _print(output_format, text, {"type": "tool_providers.list", "items": items})


def _print_tool_provider(runtime, name: str, output_format: str) -> None:
    provider = runtime.integration_catalog.provider_by_name(name)
    if not provider:
        _print(output_format, f"Unknown tool provider: {name}", {"type": "tool_provider.inspect", "found": False, "name": name})
        return
    health = provider.health().__dict__
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "provider_name": tool.provider_name,
            "provider_kind": tool.provider_kind,
            "source": tool.source,
            "schema_hash": tool.schema_hash,
            "trust_level": tool.trust_level,
            "permissions": tool.permissions,
        }
        for tool in provider.discover(refresh=False)
    ]
    payload = {"type": "tool_provider.inspect", "found": True, "health": health, "tools": tools}
    text = "\n".join(
        [
            f"Provider: {health['name']}",
            f"Kind: {health['kind']}",
            f"Status: {'ok' if health['ok'] else 'failed'}",
            f"Error: {health.get('error') or '-'}",
            "",
            "Tools:",
            *[f"- {item['name']} | {item['source']} | {item['trust_level']}" for item in tools],
        ]
    )
    _print(output_format, text, payload)


def _print_tools(runtime, task: str, output_format: str) -> None:
    selected = runtime.integration_catalog.enabled_tool_names(task, runtime.config.default_agent)
    items = runtime.integration_catalog.tool_inventory(selected)
    text = "\n".join(
        f"{item['name']} | {'enabled' if item['enabled'] else 'available'} | {item['provider_kind']} | {item['source']} | schema={item['schema_hash'] or '-'} | permissions={item['permissions']}"
        for item in items
    ) or "No tools."
    _print(output_format, text, {"type": "tools.list", "selected": selected, "items": items})


def _print_sandbox_doctor(runtime, output_format: str) -> None:
    from src.skill_plugins import doctor_sandbox

    payload = {"type": "sandbox.doctor", **doctor_sandbox(runtime.config.skill_runtime)}
    lines = [
        f"Docker enabled: {payload['docker_enabled']}",
        f"Overall: {'ok' if payload['ok'] else 'not ready'}",
        f"Allowed images: {', '.join(payload['allowed_images'])}",
    ]
    for check in payload["checks"]:
        if check["name"] == "docker_images":
            lines.append("docker_images:")
            for item in check["items"]:
                lines.append(f"- {item['image']}: {'ok' if item['ok'] else 'missing'} {item.get('stderr') or item.get('stdout') or ''}")
        else:
            lines.append(f"{check['name']}: {'ok' if check['ok'] else 'failed'} {check.get('stdout') or check.get('stderr') or ''}")
    _print(output_format, "\n".join(lines), payload)


def _print_rag_health(output_format: str) -> None:
    from src.rag_runtime import rag_health

    payload = {"type": "rag.health", **rag_health()}
    lines = [
        f"RAG: {'ok' if payload.get('ok') else 'not ready'}",
        f"Code: {payload.get('code')}",
        f"Service: {payload.get('service_base') or 'local-file-or-unconfigured'}",
        f"Knowledge Base: {payload.get('knowledge_base_id')}",
        f"Sources: {payload.get('source_count', 0)}",
        f"Latency: {payload.get('latency_ms')} ms",
    ]
    if payload.get("error"):
        lines.append(f"Error: {payload['error']}")
    if payload.get("message"):
        lines.append(f"Message: {str(payload['message'])[:500]}")
    _print(output_format, "\n".join(lines), payload)


def _print_mcp_health(runtime, output_format: str) -> None:
    statuses = [item.__dict__ for item in runtime.mcp_manager.health()]
    text = "\n".join(
        f"{item['name']} | {'enabled' if item['enabled'] else 'disabled'} | {'ok' if item['ok'] else 'failed'} | tools={','.join(item['tools']) or '-'} | {item['error']}"
        for item in statuses
    ) or "No MCP servers configured."
    _print(output_format, text, {"type": "mcp.health", "items": statuses})


def _print_mcp_list(runtime, output_format: str) -> None:
    items = []
    for name, server in runtime.config.mcp.servers.items():
        items.append({"name": name, **server.model_dump()})
    text = "\n".join(
        f"{item['name']} | {'enabled' if item['enabled'] else 'disabled'} | {item['transport']} | allow={item['allow_tools']}"
        for item in items
    ) or "No MCP servers configured."
    _print(output_format, text, {"type": "mcp.list", "items": items})


def _print_mcp_inspect(runtime, name: str, output_format: str) -> None:
    server = runtime.config.mcp.servers.get(name)
    payload = {"type": "mcp.inspect", "name": name, "found": bool(server), "server": server.model_dump() if server else None}
    text = json.dumps(payload["server"], ensure_ascii=False, indent=2) if server else f"Unknown MCP server: {name}"
    _print(output_format, text, payload)


def _print_mcp_tools(runtime, name: str, refresh: bool, output_format: str) -> None:
    if name not in runtime.config.mcp.servers:
        _print(output_format, f"Unknown MCP server: {name}", {"type": "mcp.tools", "server": name, "found": False, "items": []})
        return
    if refresh:
        runtime.mcp_manager.discover_tools(refresh=True)
    tools = runtime.mcp_manager._tool_cache.get(name) or [
        item for item in runtime.mcp_manager.configured_tool_infos() if item.server == name
    ]
    items = [item.__dict__ for item in tools]
    text = "\n".join(f"{item['local_name']} | schema={item['schema_hash']} | {item['description']}" for item in items) or "No MCP tools discovered."
    _print(output_format, text, {"type": "mcp.tools", "server": name, "found": True, "refreshed": refresh, "items": items, "error": runtime.mcp_manager._errors.get(name, "")})


def _run_mcp_tool(runtime, tool_name: str, raw_args: str, auto_approve: bool, output_format: str) -> None:
    from pathlib import Path

    from src.config import permissions_for_agent
    from src.session import new_session_id
    from src.tool_runtime import ToolRuntime, ToolRunRequest
    from src.tools.registry import TOOL_REGISTRY, ToolContext

    args, error = parse_json_object(raw_args)
    if error:
        _print(output_format, error, {"type": "mcp.tool.result", "status": "invalid_args", "error": error})
        return
    if tool_name not in TOOL_REGISTRY:
        _print(output_format, f"Unknown MCP tool: {tool_name}", {"type": "mcp.tool.result", "status": "invalid_args", "tool": tool_name})
        return
    session_id = new_session_id()
    runtime.storage.create_session(session_id, f"mcp-tool:{tool_name}", runtime.config.default_agent)
    result, approvals = ToolRuntime(Path(runtime.config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name=tool_name,
            args=args,
            context=ToolContext(session_id=session_id, step_id="mcp_cli", output_dir=Path(runtime.config.output_dir)),
            session_id=session_id,
            step_id="mcp_cli",
            enabled_tools={tool_name},
            permission_rules=permissions_for_agent(runtime.config),
            auto_approve=auto_approve,
            output_format=output_format,
            storage=runtime.storage,
        )
    )
    payload = {"type": "mcp.tool.result", "tool": tool_name, "result": result.__dict__, "approvals": approvals, "session_id": session_id}
    _print(output_format, result.display_output or result.output or result.error or "", payload)


def _run_skill_tool(runtime, tool_name: str, raw_args: str, auto_approve: bool, output_format: str) -> None:
    from pathlib import Path

    from src.config import permissions_for_agent
    from src.session import new_session_id
    from src.skills import permissions_for_skills
    from src.tool_runtime import ToolRuntime, ToolRunRequest
    from src.tools.registry import ToolContext

    args, error = parse_json_object(raw_args)
    if error:
        _print(output_format, error, {"type": "skill.tool.result", "status": "invalid_args", "error": error})
        return
    session_id = new_session_id()
    runtime.storage.create_session(session_id, f"skill-tool:{tool_name}", runtime.config.default_agent)
    enabled_skills = [name for name, skill in runtime.skill_registry.items() if tool_name in skill.tools]
    rules = permissions_for_agent(runtime.config) + permissions_for_skills(enabled_skills, runtime.skill_registry)
    result, approvals = ToolRuntime(Path(runtime.config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name=tool_name,
            args=args,
            context=ToolContext(session_id=session_id, step_id="skill_cli", output_dir=Path(runtime.config.output_dir)),
            session_id=session_id,
            step_id="skill_cli",
            enabled_tools={tool_name},
            permission_rules=rules,
            auto_approve=auto_approve,
            output_format=output_format,
            storage=runtime.storage,
        )
    )
    payload = {"type": "skill.tool.result", "tool": tool_name, "result": result.__dict__, "approvals": approvals, "session_id": session_id}
    _print(output_format, result.display_output or result.output or result.error or "", payload)


def _print_team_status(storage, team_run_id: str, output_format: str) -> None:
    run = storage.get_team_run(team_run_id)
    if not run:
        _print(output_format, f"Unknown team run: {team_run_id}", {"type": "team.status", "team_run_id": team_run_id, "found": False})
        return
    members = storage.list_team_members(team_run_id)
    tasks = storage.list_team_tasks(team_run_id)
    member_runs = storage.list_team_member_runs(team_run_id)
    text = "\n".join(
        [
            f"Team Run: {run['id']}",
            f"Team: {run['team_id']}",
            f"Status: {run['status']}",
            f"Task: {run['task']}",
            "",
            "Members:",
            *[f"- {item['name']} | {item['role']} | {item['status']} | {item['session_id']}" for item in members],
            "",
            "Tasks:",
            *[f"- {item['status']} | {item['assigned_to']} | {item['title']}" for item in tasks],
            "",
            f"Member runs: {len(member_runs)}",
        ]
    )
    _print(output_format, text, {"type": "team.status", "run": run, "members": members, "tasks": tasks, "member_runs": member_runs})


def _format_inspect_run(run_id: str, state: dict) -> str:
    return "\n".join(
        [
            f"Run: {run_id}",
            f"Task: {state.get('user_task')}",
            f"Session: {state.get('session_id')}",
            f"Complete: {state.get('is_complete')}",
            f"Acceptance: {state.get('acceptance')}",
            f"Plan: {state.get('plan')}",
            f"Tool Results: {state.get('tool_results')}",
        ]
    )


def _format_inspect_session(storage, session_id: str) -> str:
    messages = storage.list_messages(session_id)
    parts = storage.list_message_parts(session_id)
    tool_calls = storage.list_tool_calls(session_id)
    summary = storage.get_session_summary(session_id)
    return "\n".join(
        [
            f"Session: {session_id}",
            f"Messages: {len(messages)}",
            f"Parts: {len(parts)}",
            f"Tool calls: {len(tool_calls)}",
            f"Summary: {summary or '无'}",
        ]
    )


def _storage_path(app_config) -> str:
    return str(app_config.output_dir.rstrip("/\\") + "/agent.sqlite")


def _build_runtime_graph(build_graph, llm, full_model: bool, checkpoint_callback, start_at: str):
    if full_model:
        return build_graph(
            llm=llm,
            executor_llm=llm,
            verifier_llm=llm,
            summarizer_llm=llm,
            checkpoint_callback=checkpoint_callback,
            start_at=start_at,
        )
    return build_graph(
        llm=llm,
        executor_llm=llm,
        verifier_llm=None,
        summarizer_llm=None,
        checkpoint_callback=checkpoint_callback,
        start_at=start_at,
    )


def _resolve_model(cli_model: str | None, config_model: str | None) -> str | None:
    return cli_model or os.getenv("OPENAI_MODEL") or config_model


def _enabled_tools(app_config, skill_tools: list[str]) -> list[str]:
    enabled = list(skill_tools)
    for name, value in app_config.tools.items():
        if value and name not in enabled:
            enabled.append(name)
        if not value and name in enabled:
            enabled.remove(name)
    return enabled


if __name__ == "__main__":
    sys.exit(main())
