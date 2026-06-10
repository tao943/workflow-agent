from pathlib import Path
from uuid import uuid4

from src.config import AppConfig
from src.evals import (
    EvalCase,
    EvalHarness,
    EvalReport,
    EvalRunResult,
    DeepEvalEvaluator,
    RAGEvaluator,
    RuleEvaluator,
    ServiceEvaluator,
    TrajectoryEvaluator,
    build_deepeval_trace,
    _merge_scores,
)


def _eval_output_dir(name: str) -> Path:
    path = Path("outputs") / "test_evals" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_eval_case_loader_lists_suites() -> None:
    harness = EvalHarness(AppConfig(), output_dir=_eval_output_dir("list"))

    suites = {item["suite"]: item["cases"] for item in harness.list_suites()}

    assert "smoke" in suites
    assert suites["smoke"] >= 1


def test_rule_evaluator_detects_missing_section() -> None:
    case = EvalCase(id="missing_section", suite="unit", required_sections=["## Required"])
    result = EvalRunResult(case=case, status="completed", final_answer="No section", tool_calls=[], events=[], evidence=[], duration_ms=1)

    score = RuleEvaluator().evaluate(result)

    assert not score.passed
    assert "missing required section: ## Required" in score.issues


def test_trajectory_evaluator_detects_required_and_forbidden_tools() -> None:
    case = EvalCase(
        id="trajectory",
        suite="unit",
        expected_tools=["read_file"],
        forbidden_tools=["write_file"],
    )
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="ok",
        tool_calls=[{"tool_name": "write_file"}],
        events=[],
        evidence=[],
        duration_ms=1,
    )

    score = TrajectoryEvaluator().evaluate(result)

    assert not score.passed
    assert "expected tool was not called: read_file" in score.issues
    assert "forbidden tool was called: write_file" in score.issues


def test_eval_report_save_load_compare() -> None:
    output_dir = _eval_output_dir("report")
    harness = EvalHarness(AppConfig(), output_dir=output_dir)
    report = EvalReport(
        schema_version="eval-report.v2",
        eval_run_id="eval_unit_a",
        suite="unit",
        started_at="2026-06-06T00:00:00",
        duration_ms=10,
        passed=True,
        summary={"passed": 1, "total": 1},
        case_results=[],
        aggregate_metrics={"rule": 1.0},
        environment={"services": {"rag": {"ok": True}}},
    )

    harness.save_report(report)
    loaded = harness.load_report("eval_unit_a")
    comparison = harness.compare_reports("eval_unit_a", "eval_unit_a")

    assert loaded is not None
    assert loaded["eval_run_id"] == "eval_unit_a"
    assert loaded["environment"]["services"]["rag"]["ok"] is True
    assert comparison["metric_delta"]["rule"] == 0.0


def test_eval_report_external_exports() -> None:
    output_dir = _eval_output_dir("export")
    harness = EvalHarness(AppConfig(), output_dir=output_dir)
    report = EvalReport(
        schema_version="eval-report.v2",
        eval_run_id="eval_export_a",
        suite="unit",
        started_at="2026-06-07T00:00:00",
        duration_ms=10,
        passed=True,
        summary={"passed": 1, "total": 1},
        case_results=[
            {
                "case": {"id": "case_a", "suite": "unit", "task": "hello", "mode": "single", "assertions": {}},
                "run": {"final_answer": "hello result", "status": "completed", "duration_ms": 1, "tool_calls": [], "evidence": []},
                "score": {"passed": True, "quality_score": 1.0},
            }
        ],
        aggregate_metrics={"rule": 1.0},
    )
    harness.save_report(report)

    promptfoo = harness.export_report("eval_export_a", "promptfoo")
    langsmith = harness.export_report("eval_export_a", "langsmith")
    langfuse = harness.export_report("eval_export_a", "langfuse")
    openai = harness.export_report("eval_export_a", "openai")

    assert Path(promptfoo["path"]).read_text(encoding="utf-8").startswith("[")
    assert '"inputs"' in Path(langsmith["path"]).read_text(encoding="utf-8")
    assert '"metadata"' in Path(langfuse["path"]).read_text(encoding="utf-8")
    assert '"sample"' in Path(openai["path"]).read_text(encoding="utf-8")


def test_weighted_policy_allows_minor_issue_above_threshold() -> None:
    case = EvalCase(
        id="weighted",
        suite="unit",
        grading_policy="weighted",
        pass_threshold=0.5,
        required_sections=["## Required", "## Optional"],
    )
    result = EvalRunResult(case=case, status="completed", final_answer="## Required\nok", tool_calls=[], events=[], evidence=[], duration_ms=1)

    rule_score = RuleEvaluator().evaluate(result)
    merged = _merge_scores(case, [rule_score])

    assert rule_score.issues == ["missing required section: ## Optional"]
    assert merged.hard_pass
    assert merged.passed
    assert 0.5 <= merged.quality_score < 1.0


def test_forbidden_tool_is_hard_fail_even_when_weighted() -> None:
    case = EvalCase(
        id="hard_fail",
        suite="unit",
        grading_policy="weighted",
        forbidden_tools=["write_file"],
    )
    result = EvalRunResult(case=case, status="completed", final_answer="ok", tool_calls=[{"tool_name": "write_file"}], events=[], evidence=[], duration_ms=1)

    score = TrajectoryEvaluator().evaluate(result)
    merged = _merge_scores(case, [score])

    assert not merged.hard_pass
    assert not merged.passed


def test_eval_harness_runs_tool_provider_case() -> None:
    output_dir = _eval_output_dir("tool_provider")
    harness = EvalHarness(AppConfig(), output_dir=output_dir)
    case = EvalCase(
        id="tool_provider",
        suite="unit",
        task="fetch availability",
        mode="tool_provider",
        expected_tools=["mcp.fetch.fetch"],
        assertions={"contains": ["mcp.fetch.fetch"]},
    )

    result = harness._run_case("eval_unit", case, offline=True, output_format="json")
    score = TrajectoryEvaluator().evaluate(result)

    assert result.status == "completed"
    assert "mcp.fetch.fetch" in result.final_answer
    assert score.passed


def test_rag_evaluator_accepts_connected_no_evidence_honesty() -> None:
    case = EvalCase(
        id="rag_no_evidence",
        suite="unit",
        mode="rag",
        assertions={"expect_no_evidence": True},
    )
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="知识库中没有找到足够依据。",
        tool_calls=[{"tool_name": "rag_search", "metadata": {"code": "no_evidence", "sources": []}}],
        events=[],
        evidence=[],
        duration_ms=1,
    )

    score = RAGEvaluator().evaluate(result)

    assert score.passed
    assert score.metrics["no_evidence_honesty"] == 1.0


def test_rag_evaluator_accepts_unavailable_as_no_answer_evidence() -> None:
    case = EvalCase(
        id="rag_unavailable",
        suite="unit",
        mode="rag",
        assertions={"expect_no_answer_evidence": True},
    )
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="RAG 服务不可用，无法基于知识库证据回答。",
        tool_calls=[{"tool_name": "rag_search", "metadata": {"code": "service_unavailable"}}],
        events=[],
        evidence=[],
        duration_ms=1,
    )

    score = RAGEvaluator().evaluate(result)

    assert score.passed
    assert score.metrics["no_answer_evidence_honesty"] == 1.0


def test_rag_evaluator_requires_nonempty_sources() -> None:
    case = EvalCase(
        id="rag_sources",
        suite="unit",
        mode="rag",
        assertions={"require_sources": True},
    )
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="answer",
        tool_calls=[{"tool_name": "rag_search", "metadata": {"code": "ok", "sources": []}}],
        events=[],
        evidence=[],
        duration_ms=1,
    )

    score = RAGEvaluator().evaluate(result)

    assert not score.passed
    assert score.metrics["source_presence"] == 0.0


def test_service_evaluator_fails_missing_required_service() -> None:
    case = EvalCase(id="service", suite="unit", requires_services=["rag"])
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="ok",
        tool_calls=[],
        events=[],
        evidence=[],
        duration_ms=1,
        environment={"services": {"rag": {"ok": False, "code": "service_unavailable"}}},
    )

    score = ServiceEvaluator().evaluate(result)

    assert not score.passed
    assert not score.hard_pass
    assert score.metrics["service_rag"] == 0.0


def test_deepeval_trace_adapter_builds_agent_and_tool_spans() -> None:
    case = EvalCase(id="trace", suite="unit", task="calculate", expected_tools=["calculator"])
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="done",
        tool_calls=[{"tool_name": "calculator", "input": {"input": "1+1"}, "result": "2", "status": "success"}],
        events=[{"type": "planner.completed", "data": {"steps": 1}}],
        evidence=[],
        duration_ms=10,
        run_id="run_a",
    )

    trace = build_deepeval_trace(result)

    assert trace["case_id"] == "trace"
    assert any(span["type"] == "agent" for span in trace["spans"])
    assert any(span["type"] == "tool" and span["name"] == "calculator" for span in trace["spans"])


def test_deepeval_evaluator_skips_when_disabled() -> None:
    case = EvalCase(id="deepeval", suite="unit")
    result = EvalRunResult(case=case, status="completed", final_answer="ok", tool_calls=[], events=[], evidence=[], duration_ms=1)

    config = AppConfig()
    config.evals.enable_deepeval = False
    score = DeepEvalEvaluator(config).evaluate(result)

    assert score.passed
    assert score.metrics["deepeval_skipped"] == 1.0


def test_deepeval_evaluator_reports_native_mode_when_enabled() -> None:
    case = EvalCase(id="deepeval_native", suite="unit", task="calculate", expected_tools=["calculator"])
    result = EvalRunResult(
        case=case,
        status="completed",
        final_answer="done",
        tool_calls=[{"tool_name": "calculator", "input": {"input": "1+1"}, "result": "2", "status": "success"}],
        events=[],
        evidence=[],
        duration_ms=1,
    )
    config = AppConfig()
    config.evals.deepeval_mode = "native"

    score = DeepEvalEvaluator(config).evaluate(result)

    serialized = "\n".join(score.artifacts)
    assert score.passed
    assert "native" in serialized or "deepeval_dependency_missing" in score.metrics
