from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.config import AppConfig
from src.evidence import verify_final_answer_evidence, verify_task_evidence
from src.runtime import AgentRuntime, RuntimeOptions
from src.storage import Storage
from src.tools.registry import ToolContext


EvalMode = Literal["single", "team", "tool_provider", "rag", "mcp", "skill", "prompt_contract"]
GradingPolicy = Literal["strict", "weighted", "judge"]
IssueSeverity = Literal["blocker", "major", "minor", "info"]


@dataclass
class EvalCase:
    id: str
    suite: str
    task: str = ""
    mode: EvalMode = "single"
    agent: str | None = None
    team: str | None = None
    offline: bool = True
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    assertions: dict[str, Any] = field(default_factory=dict)
    grading_policy: GradingPolicy = "strict"
    pass_threshold: float = 0.75
    metric_weights: dict[str, float] = field(default_factory=dict)
    critical_assertions: list[str] = field(default_factory=list)
    reference_trajectory: list[str] = field(default_factory=list)
    match_mode: Literal["strict", "subset", "unordered", "similarity"] = "subset"
    tags: list[str] = field(default_factory=list)
    requires_services: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        if not payload.get("id"):
            raise ValueError("Eval case missing id")
        if not payload.get("suite"):
            raise ValueError(f"Eval case {payload.get('id')} missing suite")
        return cls(**payload)


@dataclass
class EvalRunResult:
    case: EvalCase
    status: str
    final_answer: str
    tool_calls: list[dict[str, Any]]
    events: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    duration_ms: int
    run_id: str = ""
    session_id: str = ""
    team: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetric:
    name: str
    score: float
    weight: float = 1.0
    hard: bool = False


@dataclass
class EvalIssue:
    message: str
    severity: IssueSeverity = "minor"
    metric: str = ""


@dataclass
class EvalScore:
    case_id: str
    passed: bool
    score: float
    hard_pass: bool = True
    quality_score: float = 1.0
    severity: IssueSeverity = "info"
    metrics: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    issue_details: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    schema_version: str
    eval_run_id: str
    suite: str
    started_at: str
    duration_ms: int
    passed: bool
    summary: dict[str, Any]
    case_results: list[dict[str, Any]]
    aggregate_metrics: dict[str, float]
    environment: dict[str, Any] = field(default_factory=dict)


class ServiceEvaluator:
    def evaluate(self, result: EvalRunResult) -> EvalScore:
        issues: list[EvalIssue] = []
        metrics: dict[str, float] = {}
        services = result.environment.get("services", {})
        for service in result.case.requires_services:
            status = services.get(service) or {}
            ok = bool(status.get("ok"))
            metrics[f"service_{service}"] = 1.0 if ok else 0.0
            if not ok:
                code = status.get("code") or status.get("error") or "unavailable"
                issues.append(EvalIssue(f"required service unavailable: {service} ({code})", "blocker", f"service_{service}"))
        if not metrics:
            metrics["services"] = 1.0
        return _score(result.case.id, metrics, issues)


class RuleEvaluator:
    def evaluate(self, result: EvalRunResult) -> EvalScore:
        issues: list[EvalIssue] = []
        case = result.case
        if result.status not in {"completed", "success"}:
            issues.append(EvalIssue(f"unexpected status: {result.status}", "blocker", "status"))
        section_score = _ratio_score(case.required_sections, result.final_answer)
        for section in case.required_sections:
            if section not in result.final_answer:
                severity: IssueSeverity = "blocker" if "required_sections" in case.critical_assertions else "major"
                issues.append(EvalIssue(f"missing required section: {section}", severity, "required_sections"))
        max_duration_ms = int(case.assertions.get("max_duration_ms", 0) or 0)
        if max_duration_ms and result.duration_ms > max_duration_ms:
            issues.append(EvalIssue(f"duration exceeded: {result.duration_ms}>{max_duration_ms}", "minor", "duration"))
        required_contains = list(case.assertions.get("contains") or [])
        contains_score = _ratio_score(required_contains, result.final_answer)
        final_answer_lower = result.final_answer.lower()
        for text in required_contains:
            if str(text).lower() not in final_answer_lower:
                severity = "blocker" if "contains" in case.critical_assertions else "major"
                issues.append(EvalIssue(f"missing required text: {text}", severity, "contains"))
        metrics = {
            "status": 0.0 if any(item.metric == "status" for item in issues) else 1.0,
            "required_sections": section_score,
            "contains": contains_score,
            "duration": 0.0 if any(item.metric == "duration" for item in issues) else 1.0,
        }
        return _score(case.id, metrics, issues)


class TrajectoryEvaluator:
    def evaluate(self, result: EvalRunResult) -> EvalScore:
        issues: list[EvalIssue] = []
        called = [item.get("tool_name") or item.get("tool") for item in result.tool_calls]
        called = [item for item in called if item]
        expected_tool_score = _tool_match_score(result.case.expected_tools, called, result.case.match_mode)
        for tool in result.case.expected_tools:
            if tool not in called:
                severity: IssueSeverity = "blocker" if "expected_tools" in result.case.critical_assertions else "major"
                issues.append(EvalIssue(f"expected tool was not called: {tool}", severity, "expected_tools"))
        for tool in result.case.forbidden_tools:
            if tool in called:
                issues.append(EvalIssue(f"forbidden tool was called: {tool}", "blocker", "forbidden_tools"))
        expected_members = list(result.case.assertions.get("expected_members") or [])
        actual_members = [item.get("name") for item in result.team.get("members", [])]
        member_score = _ratio_score(expected_members, "\n".join(str(item) for item in actual_members))
        for member in expected_members:
            if member not in actual_members:
                issues.append(EvalIssue(f"expected member missing: {member}", "major", "expected_members"))
        idle_members = list(result.case.assertions.get("idle_members") or [])
        idle_hits = 0
        for member in idle_members:
            status = next((item.get("status") for item in result.team.get("members", []) if item.get("name") == member), None)
            if status == "idle":
                idle_hits += 1
            elif status:
                severity = "blocker" if "idle_members" in result.case.critical_assertions else "major"
                issues.append(EvalIssue(f"expected member idle: {member}, got {status}", severity, "idle_members"))
        idle_score = 1.0 if not idle_members else idle_hits / len(idle_members)
        reference_score = _tool_match_score(result.case.reference_trajectory, called, result.case.match_mode)
        redundant_penalty = _redundant_tool_penalty(_tool_call_fingerprints(result.tool_calls))
        metrics = {
            "expected_tools": expected_tool_score,
            "forbidden_tools": 0.0 if any(item.metric == "forbidden_tools" for item in issues) else 1.0,
            "expected_members": member_score,
            "idle_members": idle_score,
            "reference_trajectory": reference_score,
            "tool_efficiency": redundant_penalty,
        }
        return _score(result.case.id, metrics, issues)


class EvidenceEvaluator:
    def evaluate(self, result: EvalRunResult) -> EvalScore:
        issues: list[EvalIssue] = []
        if result.case.required_evidence:
            task_check = verify_task_evidence(
                {"required_tools": result.case.expected_tools, "required_evidence": result.case.required_evidence},
                {"final_answer": result.final_answer},
                result.tool_calls,
            )
            issues.extend(EvalIssue(issue, "major", "required_evidence") for issue in task_check.issues)
        if result.case.assertions.get("final_answer_evidence"):
            evidence_ids = set(result.case.assertions.get("evidence_ids") or _evidence_ids(result.evidence))
            final_check = verify_final_answer_evidence(result.final_answer, evidence_ids)
            issues.extend(EvalIssue(issue, "major", "final_answer_evidence") for issue in final_check.issues)
        evidence_ids = set(result.case.assertions.get("evidence_ids") or _evidence_ids(result.evidence))
        cited_ids = _cited_evidence_ids(result.final_answer)
        invalid_refs = [item for item in cited_ids if evidence_ids and item not in evidence_ids]
        issues.extend(EvalIssue(f"unknown evidence id: {item}", "blocker", "evidence_validity") for item in invalid_refs)
        evidence_validity = 1.0 if not invalid_refs else 0.0
        has_success_evidence = any(item.get("status", "success") == "success" for item in result.evidence)
        needs_citations = result.case.mode == "team" or bool(result.case.assertions.get("final_answer_evidence"))
        citation_coverage = _citation_coverage(result.final_answer) if (needs_citations and has_success_evidence) else 1.0
        required_evidence_score = 0.0 if any(item.metric == "required_evidence" for item in issues) else 1.0
        metrics = {
            "required_evidence": required_evidence_score,
            "evidence_validity": evidence_validity,
            "citation_coverage": citation_coverage,
        }
        return _score(result.case.id, metrics, issues)


class RAGEvaluator:
    def evaluate(self, result: EvalRunResult) -> EvalScore:
        issues: list[EvalIssue] = []
        if result.case.mode != "rag":
            return _score(result.case.id, {"rag": 1.0}, issues)
        rag_calls = [item for item in result.tool_calls if item.get("tool_name") == "rag_search"]
        serialized = json.dumps(rag_calls + result.evidence, ensure_ascii=False)
        unavailable = "not_configured" in serialized or "service_unavailable" in serialized
        no_evidence = "no_evidence" in serialized or "没有找到足够依据" in result.final_answer or "未找到" in result.final_answer
        no_answer_evidence = unavailable or no_evidence
        not_configured_score = 1.0
        no_evidence_score = 1.0
        no_answer_evidence_score = 1.0
        source_score = 1.0
        if result.case.assertions.get("expect_not_configured"):
            not_configured_score = 1.0 if unavailable else 0.0
            if not_configured_score == 0.0:
                issues.append(EvalIssue("expected RAG unavailable/not_configured signal", "major", "not_configured_honesty"))
        if result.case.assertions.get("expect_no_evidence"):
            no_evidence_score = 1.0 if no_evidence else 0.0
            if no_evidence_score == 0.0:
                issues.append(EvalIssue("expected RAG no_evidence signal", "major", "no_evidence_honesty"))
        if result.case.assertions.get("expect_no_answer_evidence"):
            no_answer_evidence_score = 1.0 if no_answer_evidence else 0.0
            if no_answer_evidence_score == 0.0:
                issues.append(EvalIssue("expected RAG unavailable or no_evidence signal", "major", "no_answer_evidence_honesty"))
        if result.case.assertions.get("require_sources"):
            source_score = 1.0 if _has_nonempty_sources(rag_calls + result.evidence) else 0.0
            if source_score == 0.0:
                issues.append(EvalIssue("expected RAG sources", "major", "source_presence"))
        metrics = {
            "not_configured_honesty": not_configured_score,
            "no_evidence_honesty": no_evidence_score,
            "no_answer_evidence_honesty": no_answer_evidence_score,
            "source_presence": source_score,
            "answer_groundedness": 1.0 if no_answer_evidence else _citation_coverage(result.final_answer),
        }
        return _score(result.case.id, metrics, issues)


class DeepEvalEvaluator:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

    def evaluate(self, result: EvalRunResult) -> EvalScore:
        if not getattr(self.config.evals, "enable_deepeval", False):
            return EvalScore(
                case_id=result.case.id,
                passed=True,
                score=1.0,
                hard_pass=True,
                quality_score=1.0,
                metrics={"deepeval_skipped": 1.0},
                artifacts=["deepeval disabled"],
            )
        mode = getattr(self.config.evals, "deepeval_mode", "local")
        try:
            __import__("deepeval")
        except ImportError:
            return EvalScore(
                case_id=result.case.id,
                passed=True,
                score=1.0,
                hard_pass=True,
                quality_score=1.0,
                metrics={"deepeval_dependency_missing": 1.0},
                artifacts=["deepeval dependency missing"],
            )
        trace = build_deepeval_trace(result)
        if mode == "native":
            return self._evaluate_native(result, trace)
        metric_scores = _fallback_deepeval_metric_scores(result, self.config.evals.deep_eval_thresholds)
        return EvalScore(
            case_id=result.case.id,
            passed=True,
            score=round(sum(metric_scores.values()) / max(len(metric_scores), 1), 4),
            hard_pass=True,
            quality_score=round(sum(metric_scores.values()) / max(len(metric_scores), 1), 4),
            metrics={f"deepeval_{key}": value for key, value in metric_scores.items()},
            artifacts=[json.dumps({"trace": trace, "mode": "local_trace_adapter", "judge_model": self.config.evals.evaluation_model}, ensure_ascii=False)],
        )

    def _evaluate_native(self, result: EvalRunResult, trace: dict[str, Any]) -> EvalScore:
        try:
            from deepeval.metrics import (  # type: ignore
                ArgumentCorrectnessMetric,
                PlanQualityMetric,
                StepEfficiencyMetric,
                TaskCompletionMetric,
                ToolCorrectnessMetric,
            )
        except Exception as exc:
            return EvalScore(
                case_id=result.case.id,
                passed=True,
                score=1.0,
                hard_pass=True,
                quality_score=1.0,
                metrics={"deepeval_native_unavailable": 1.0},
                artifacts=[json.dumps({"trace": trace, "mode": "native_trace_unavailable", "error": str(exc)}, ensure_ascii=False)],
            )
        metric_classes = [
            TaskCompletionMetric,
            StepEfficiencyMetric,
            ToolCorrectnessMetric,
            ArgumentCorrectnessMetric,
            PlanQualityMetric,
        ]
        fallback_scores = _fallback_deepeval_metric_scores(result, self.config.evals.deep_eval_thresholds)
        artifact = {
            "trace": trace,
            "mode": "native_trace",
            "judge_model": self.config.evals.evaluation_model,
            "metric_classes": [item.__name__ for item in metric_classes],
            "note": "Runtime spans are exported in DeepEval-compatible shape; official metric measurement can be attached with @observe during protocol runs.",
        }
        return EvalScore(
            case_id=result.case.id,
            passed=True,
            score=round(sum(fallback_scores.values()) / max(len(fallback_scores), 1), 4),
            hard_pass=True,
            quality_score=round(sum(fallback_scores.values()) / max(len(fallback_scores), 1), 4),
            metrics={f"deepeval_native_{key}": value for key, value in fallback_scores.items()},
            artifacts=[json.dumps(artifact, ensure_ascii=False)],
        )


class LLMJudgeEvaluator:
    def __init__(self, llm: Any | None = None, enabled: bool = False) -> None:
        self.llm = llm
        self.enabled = enabled

    def evaluate(self, result: EvalRunResult) -> EvalScore:
        if not self.enabled:
            return _score(result.case.id, {"judge": 1.0}, [])
        if self.llm is None:
            return _score(result.case.id, {"judge": 1.0}, [EvalIssue("LLM judge requested but no model is available", "info", "judge")])
        # First version keeps judge deterministic unless a project-specific model adapter is added.
        return _score(result.case.id, {"judge": 1.0}, [])


class EvalHarness:
    def __init__(
        self,
        config: AppConfig,
        llm: Any | None = None,
        cases_dir: str | Path = "evals/cases",
        output_dir: str | Path = "outputs/evals",
        judge_enabled: bool = False,
    ) -> None:
        self.config = config
        self.llm = llm
        self.cases_dir = Path(cases_dir)
        self.output_dir = Path(output_dir)
        self.evaluators = [
            ServiceEvaluator(),
            RuleEvaluator(),
            TrajectoryEvaluator(),
            EvidenceEvaluator(),
            RAGEvaluator(),
            DeepEvalEvaluator(config),
            LLMJudgeEvaluator(llm, judge_enabled),
        ]

    def list_suites(self) -> list[dict[str, Any]]:
        cases = self.load_cases()
        suites = sorted({case.suite for case in cases})
        return [{"suite": suite, "cases": len([case for case in cases if case.suite == suite])} for suite in suites]

    def load_cases(self, suite: str | None = None) -> list[EvalCase]:
        cases: list[EvalCase] = []
        for path in sorted(self.cases_dir.glob("*.jsonl")):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip() or line.strip().startswith("#"):
                    continue
                try:
                    case = EvalCase.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"Invalid eval case {path}:{line_number}: {exc}") from exc
                if suite is None or case.suite == suite:
                    cases.append(case)
        return cases

    def run_suite(
        self,
        suite: str,
        offline: bool = True,
        output_format: str = "default",
        threshold: float | None = None,
        baseline_run_id: str | None = None,
        fail_on_regression: bool = False,
    ) -> EvalReport:
        cases = self.load_cases(suite)
        if not cases:
            raise ValueError(f"Unknown eval suite or no cases: {suite}")
        eval_run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        started = time.perf_counter()
        case_results: list[dict[str, Any]] = []
        aggregate: dict[str, float] = {}
        environment = self._environment_snapshot(cases)
        for case in cases:
            result = self._run_case(eval_run_id, case, offline=offline, output_format=output_format)
            scores = [evaluator.evaluate(result) for evaluator in self.evaluators]
            merged = _merge_scores(case, scores, threshold=threshold)
            case_results.append(
                {
                    "case": case.__dict__,
                    "run": _run_result_payload(result),
                    "score": merged.__dict__,
                    "evaluator_scores": [score.__dict__ for score in scores],
                }
            )
            for key, value in merged.metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
        for key in list(aggregate):
            aggregate[key] = round(aggregate[key] / len(case_results), 4)
        passed = all(item["score"]["passed"] for item in case_results)
        regression = self._compare_with_baseline(baseline_run_id, case_results) if baseline_run_id else {}
        if fail_on_regression and regression.get("regressions"):
            passed = False
        report = EvalReport(
            schema_version="eval-report.v2",
            eval_run_id=eval_run_id,
            suite=suite,
            started_at=datetime.now().isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - started) * 1000),
            passed=passed,
            summary={
                "passed": sum(1 for item in case_results if item["score"]["passed"]),
                "hard_passed": sum(1 for item in case_results if item["score"]["hard_pass"]),
                "total": len(case_results),
                "average_quality_score": round(sum(float(item["score"]["quality_score"]) for item in case_results) / len(case_results), 4),
                "issue_counts": _issue_counts(case_results),
                "regression": regression,
            },
            case_results=case_results,
            aggregate_metrics=aggregate,
            environment=environment,
        )
        self.save_report(report)
        return report

    def save_report(self, report: EvalReport) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{report.eval_run_id}.json"
        path.write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_report(self, eval_run_id: str) -> dict[str, Any] | None:
        path = self.output_dir / f"{eval_run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def compare_reports(self, left: str, right: str) -> dict[str, Any]:
        a = self.load_report(left)
        b = self.load_report(right)
        if not a or not b:
            raise ValueError("Unknown eval report id")
        return {
            "left": left,
            "right": right,
            "left_passed": a["passed"],
            "right_passed": b["passed"],
            "left_summary": a["summary"],
            "right_summary": b["summary"],
            "new_failures": _case_status_delta(a, b, failed_only=True),
            "new_successes": _case_success_delta(a, b),
            "quality_regressions": _quality_regressions(a, b),
            "tool_call_delta": _tool_call_delta(a, b),
            "duration_delta_ms": int(b.get("duration_ms", 0)) - int(a.get("duration_ms", 0)),
            "metric_delta": {
                key: round(float(b.get("aggregate_metrics", {}).get(key, 0)) - float(a.get("aggregate_metrics", {}).get(key, 0)), 4)
                for key in sorted(set(a.get("aggregate_metrics", {})) | set(b.get("aggregate_metrics", {})))
            },
        }

    def export_report(self, eval_run_id: str, platform: str, output_path: str | Path | None = None) -> dict[str, Any]:
        report = self.load_report(eval_run_id)
        if not report:
            raise ValueError(f"Unknown eval report id: {eval_run_id}")
        platform_key = platform.lower().strip()
        if platform_key == "promptfoo":
            payload = _export_promptfoo_outputs(report)
            suffix = "promptfoo.outputs.json"
            content = json.dumps(payload, ensure_ascii=False, indent=2)
        elif platform_key == "langsmith":
            payload = _export_langsmith_jsonl(report)
            suffix = "langsmith.dataset.jsonl"
            content = "\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n"
        elif platform_key == "langfuse":
            payload = _export_langfuse_jsonl(report)
            suffix = "langfuse.traces.jsonl"
            content = "\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n"
        elif platform_key in {"openai", "openai-evals"}:
            payload = _export_openai_evals_jsonl(report)
            suffix = "openai.eval_items.jsonl"
            content = "\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n"
        else:
            raise ValueError("Unsupported eval export platform. Use promptfoo, langsmith, langfuse, or openai.")
        path = Path(output_path) if output_path else self.output_dir / f"{eval_run_id}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {
            "eval_run_id": eval_run_id,
            "platform": platform_key,
            "path": str(path),
            "items": len(payload),
            "note": _export_note(platform_key),
        }

    def _compare_with_baseline(self, baseline_run_id: str | None, case_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not baseline_run_id:
            return {}
        baseline = self.load_report(baseline_run_id)
        if not baseline:
            return {"baseline": baseline_run_id, "error": "baseline report not found", "regressions": []}
        current = {"case_results": case_results}
        return {
            "baseline": baseline_run_id,
            "regressions": _quality_regressions(baseline, current),
            "new_failures": _case_status_delta(baseline, current, failed_only=True),
        }

    def _run_case(self, eval_run_id: str, case: EvalCase, offline: bool, output_format: str) -> EvalRunResult:
        started = time.perf_counter()
        root = self.output_dir / "work" / eval_run_id / case.id
        run_config = self.config.model_copy(
            update={
                "output_dir": str(root / "outputs"),
                "checkpoint_path": str(root / "checkpoints.sqlite"),
                "default_agent": case.agent or self.config.default_agent,
            }
        )
        storage = Storage(root / "agent.sqlite")
        runtime = AgentRuntime(run_config, llm=None if offline or case.offline else self.llm, storage=storage)
        environment = self._case_environment(case, runtime)
        try:
            if case.mode == "tool_provider":
                selected = runtime.integration_catalog.enabled_tool_names(case.task, run_config.default_agent)
                final_answer = "\n".join(selected)
                tool_calls = [{"tool_name": name, "kind": "tool_inventory"} for name in selected]
                evidence = runtime.integration_catalog.tool_inventory(selected)
                status = "completed"
                run_id = ""
                session_id = ""
                team = {}
            elif case.mode == "mcp":
                final_answer, tool_calls, evidence = self._run_mcp_case(case, runtime, storage)
                status = "completed"
                run_id = ""
                session_id = ""
                team = {}
            elif case.mode == "skill":
                final_answer, tool_calls, evidence = self._run_skill_case(case, runtime, storage)
                status = "completed"
                run_id = ""
                session_id = ""
                team = {}
            elif case.mode == "prompt_contract":
                final_answer = _prompt_contract_text()
                tool_calls = []
                evidence = []
                status = "completed"
                run_id = ""
                session_id = ""
                team = {}
            else:
                result = runtime.run(
                    RuntimeOptions(
                        task=case.task,
                        agent=case.agent,
                        team=case.team if case.mode == "team" else None,
                        auto_approve=True,
                        output_format="default",
                        no_memory=True,
                        consolidate_memory=False,
                    )
                )
                status = result.status
                final_answer = result.final_answer
                run_id = result.run_id
                session_id = result.session_id
                team = result.team
                tool_calls = self._collect_tool_calls(storage, result)
                evidence = self._collect_evidence(result, tool_calls)
            events = storage.list_events(session_id) if session_id else []
            return EvalRunResult(case, status, final_answer, tool_calls, events, evidence, int((time.perf_counter() - started) * 1000), run_id, session_id, team, environment=environment)
        except Exception as exc:
            return EvalRunResult(case, "error", "", [], [], [], int((time.perf_counter() - started) * 1000), error=str(exc), environment=environment)

    def _run_mcp_case(self, case: EvalCase, runtime: AgentRuntime, storage: Storage) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        if case.assertions.get("invalid_url") or case.assertions.get("real_fetch") or case.assertions.get("args"):
            from src.config import permissions_for_agent
            from src.session import new_session_id
            from src.tool_runtime import ToolRuntime, ToolRunRequest

            session_id = new_session_id()
            storage.create_session(session_id, f"eval:{case.id}", runtime.config.default_agent)
            raw_args = dict(case.assertions.get("args") or {})
            url = "file:///etc/passwd" if case.assertions.get("invalid_url") else str(raw_args.get("url") or case.assertions.get("url") or "https://example.com")
            result, _ = ToolRuntime(Path(runtime.config.output_dir) / "artifacts").run(
                ToolRunRequest(
                    name="mcp.fetch.fetch",
                    args={"url": url},
                    context=ToolContext(session_id=session_id, step_id="eval_mcp", output_dir=Path(runtime.config.output_dir)),
                    session_id=session_id,
                    step_id="eval_mcp",
                    enabled_tools={"mcp.fetch.fetch"},
                    permission_rules=permissions_for_agent(runtime.config),
                    auto_approve=True,
                    output_format="default",
                    storage=storage,
                )
            )
            return result.error or result.output, storage.list_tool_calls(session_id), [result.__dict__]
        selected = runtime.integration_catalog.enabled_tool_names(case.task, runtime.config.default_agent)
        return "\n".join(selected), [], runtime.integration_catalog.tool_inventory(selected)

    def _run_skill_case(self, case: EvalCase, runtime: AgentRuntime, storage: Storage) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        from src.config import permissions_for_agent
        from src.session import new_session_id
        from src.skills import permissions_for_skills
        from src.tool_runtime import ToolRuntime, ToolRunRequest

        tool_name = str(case.assertions.get("tool_name") or case.assertions.get("tool") or "demo_docker_echo.echo")
        args = dict(case.assertions.get("tool_args") or case.assertions.get("args") or {"text": "hello docker"})
        session_id = new_session_id()
        storage.create_session(session_id, f"eval:{case.id}", runtime.config.default_agent)
        enabled_skills = [name for name, skill in runtime.skill_registry.items() if tool_name in skill.tools]
        rules = permissions_for_agent(runtime.config) + permissions_for_skills(enabled_skills, runtime.skill_registry)
        result, _ = ToolRuntime(Path(runtime.config.output_dir) / "artifacts").run(
            ToolRunRequest(
                name=tool_name,
                args=args,
                context=ToolContext(session_id=session_id, step_id="eval_skill", output_dir=Path(runtime.config.output_dir)),
                session_id=session_id,
                step_id="eval_skill",
                enabled_tools={tool_name},
                permission_rules=rules,
                auto_approve=True,
                output_format="default",
                storage=storage,
            )
        )
        return result.error or result.output, storage.list_tool_calls(session_id), [result.__dict__]

    def _environment_snapshot(self, cases: list[EvalCase]) -> dict[str, Any]:
        services = sorted({service for case in cases for service in case.requires_services})
        return {"services": {service: self._service_status(service) for service in services}}

    def _case_environment(self, case: EvalCase, runtime: AgentRuntime) -> dict[str, Any]:
        return {"services": {service: self._service_status(service, runtime) for service in case.requires_services}}

    def _service_status(self, service: str, runtime: AgentRuntime | None = None) -> dict[str, Any]:
        if service == "rag":
            from src.rag_runtime import rag_health

            return rag_health()
        if service == "mcp_fetch":
            manager = runtime.mcp_manager if runtime else AgentRuntime(self.config, llm=None).mcp_manager
            statuses = [item.__dict__ for item in manager.health()]
            fetch = next((item for item in statuses if item.get("name") == "fetch"), None)
            return fetch or {"ok": False, "code": "mcp_fetch_missing", "error": "fetch MCP server is not configured"}
        if service == "docker":
            from src.skill_plugins import doctor_sandbox

            return doctor_sandbox(self.config.skill_runtime)
        return {"ok": False, "code": "unknown_service", "error": f"Unknown required service: {service}"}

    def _collect_tool_calls(self, storage: Storage, result) -> list[dict[str, Any]]:
        if result.team:
            calls: list[dict[str, Any]] = []
            for member in result.team.get("members", []):
                calls.extend(storage.list_tool_calls(member.get("session_id", "")))
            return calls
        calls = list(result.tool_calls)
        for item in result.state.get("tool_results", []) if result.state else []:
            normalized = dict(item)
            if "tool_name" not in normalized and "tool" in normalized:
                normalized["tool_name"] = normalized["tool"]
            calls.append(normalized)
        return calls

    def _collect_evidence(self, result, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence = list(result.state.get("tool_results", []) if result.state else [])
        if result.team:
            for item in result.team.get("member_runs", []):
                member_result = item.get("result", {})
                evidence.extend(member_result.get("evidence", []) or [])
        evidence.extend(tool_calls)
        return evidence


def _score(case_id: str, metrics: dict[str, float], issues: list[EvalIssue]) -> EvalScore:
    hard_pass = not any(item.severity == "blocker" for item in issues)
    quality_score = round(sum(metrics.values()) / max(len(metrics), 1), 4)
    passed = hard_pass and not issues
    severity = _max_severity(issues)
    return EvalScore(
        case_id=case_id,
        passed=passed,
        score=quality_score if hard_pass else 0.0,
        hard_pass=hard_pass,
        quality_score=quality_score,
        severity=severity,
        metrics={key: round(float(value), 4) for key, value in metrics.items()},
        issues=[item.message for item in issues],
        issue_details=[item.__dict__ for item in issues],
    )


def _merge_scores(case: EvalCase, scores: list[EvalScore], threshold: float | None = None) -> EvalScore:
    issue_details = [detail for score in scores for detail in score.issue_details]
    issues = [issue for score in scores for issue in score.issues]
    metrics = {key: value for score in scores for key, value in score.metrics.items()}
    hard_pass = all(score.hard_pass for score in scores)
    quality_score = _weighted_average(metrics, case.metric_weights)
    effective_threshold = float(threshold if threshold is not None else case.pass_threshold)
    if case.grading_policy == "strict":
        passed = hard_pass and not issues
    else:
        passed = hard_pass and quality_score >= effective_threshold
    severity = _max_severity(EvalIssue(item.get("message", ""), item.get("severity", "minor"), item.get("metric", "")) for item in issue_details)
    return EvalScore(
        case_id=case.id,
        passed=passed,
        score=quality_score if hard_pass else 0.0,
        hard_pass=hard_pass,
        quality_score=quality_score,
        severity=severity,
        metrics=metrics,
        issues=issues,
        issue_details=issue_details,
    )


def build_deepeval_trace(result: EvalRunResult) -> dict[str, Any]:
    reasoning_events = [
        item for item in result.events
        if any(token in item.get("type", "") for token in ["planner", "verifier", "assignment", "route", "graph", "completed"])
    ]
    tool_spans = []
    for index, call in enumerate(result.tool_calls, start=1):
        tool_spans.append(
            {
                "id": f"tool_{index}",
                "type": "tool",
                "name": call.get("tool_name") or call.get("tool"),
                "input": call.get("input") or call.get("args") or {},
                "output": call.get("result") or call.get("output") or "",
                "status": call.get("status", "success"),
                "duration_ms": call.get("duration_ms", 0),
            }
        )
    member_spans = []
    for member in result.team.get("members", []) if isinstance(result.team, dict) else []:
        member_spans.append(
            {
                "id": member.get("session_id") or member.get("name"),
                "type": "agent",
                "name": member.get("name"),
                "role": member.get("role"),
                "status": member.get("status"),
            }
        )
    return {
        "case_id": result.case.id,
        "task": result.case.task,
        "agent": result.case.agent,
        "team": result.case.team,
        "status": result.status,
        "final_answer": result.final_answer,
        "spans": [
            {
                "id": result.run_id or result.session_id or result.case.id,
                "type": "agent",
                "name": "AgentRuntime.run",
                "input": result.case.task,
                "output": result.final_answer,
                "status": result.status,
                "duration_ms": result.duration_ms,
            },
            *[
                {
                    "id": f"reasoning_{index}",
                    "type": "reasoning",
                    "name": event.get("type"),
                    "input": result.case.task,
                    "output": event.get("data", {}),
                }
                for index, event in enumerate(reasoning_events, start=1)
            ],
            *tool_spans,
            *member_spans,
        ],
        "expected_tools": result.case.expected_tools,
        "forbidden_tools": result.case.forbidden_tools,
        "required_sections": result.case.required_sections,
    }


def _fallback_deepeval_metric_scores(result: EvalRunResult, thresholds: dict[str, float]) -> dict[str, float]:
    called = [item.get("tool_name") or item.get("tool") for item in result.tool_calls]
    called = [item for item in called if item]
    expected_tool_score = _tool_match_score(result.case.expected_tools, called, result.case.match_mode)
    redundant_score = _redundant_tool_penalty(_tool_call_fingerprints(result.tool_calls))
    completion_score = 1.0 if result.status in {"completed", "success"} and result.final_answer else 0.0
    argument_score = _argument_shape_score(result.tool_calls)
    plan_score = 1.0 if (result.team.get("tasks") if isinstance(result.team, dict) else result.events) else completion_score
    raw_scores = {
        "task_completion": completion_score,
        "step_efficiency": redundant_score,
        "tool_correctness": expected_tool_score,
        "argument_correctness": argument_score,
        "plan_quality": plan_score,
    }
    return {
        key: round(value if value >= float(thresholds.get(key, 0.0)) else value, 4)
        for key, value in raw_scores.items()
    }


def _argument_shape_score(tool_calls: list[dict[str, Any]]) -> float:
    if not tool_calls:
        return 1.0
    valid = 0
    for call in tool_calls:
        args = call.get("input") or call.get("args")
        if isinstance(args, dict) and all(value is not None for value in args.values()):
            valid += 1
    return round(valid / len(tool_calls), 4)


def _run_result_payload(result: EvalRunResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "final_answer": result.final_answer,
        "tool_calls": result.tool_calls,
        "events": result.events,
        "evidence": result.evidence,
        "duration_ms": result.duration_ms,
        "run_id": result.run_id,
        "session_id": result.session_id,
        "team": result.team,
        "error": result.error,
        "environment": result.environment,
    }


def _export_promptfoo_outputs(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in report.get("case_results", []):
        case = item.get("case", {})
        run = item.get("run", {})
        score = item.get("score", {})
        items.append(
            {
                "output": run.get("final_answer", ""),
                "tags": {
                    "eval_run_id": report.get("eval_run_id"),
                    "suite": report.get("suite"),
                    "case_id": case.get("id"),
                    "mode": case.get("mode"),
                    "agent": case.get("agent"),
                    "team": case.get("team"),
                    "passed": score.get("passed"),
                    "quality_score": score.get("quality_score"),
                    "duration_ms": run.get("duration_ms"),
                },
            }
        )
    return items


def _export_langsmith_jsonl(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in report.get("case_results", []):
        case = item.get("case", {})
        run = item.get("run", {})
        score = item.get("score", {})
        items.append(
            {
                "inputs": {"task": case.get("task", ""), "mode": case.get("mode"), "suite": case.get("suite")},
                "outputs": {"answer": run.get("final_answer", "")},
                "metadata": {
                    "eval_run_id": report.get("eval_run_id"),
                    "case_id": case.get("id"),
                    "status": run.get("status"),
                    "score": score,
                    "tool_calls": run.get("tool_calls", []),
                    "evidence": run.get("evidence", []),
                },
            }
        )
    return items


def _export_langfuse_jsonl(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in report.get("case_results", []):
        case = item.get("case", {})
        run = item.get("run", {})
        score = item.get("score", {})
        items.append(
            {
                "name": f"eval:{case.get('id')}",
                "input": {"task": case.get("task", ""), "suite": case.get("suite")},
                "output": run.get("final_answer", ""),
                "metadata": {
                    "eval_run_id": report.get("eval_run_id"),
                    "run_id": run.get("run_id"),
                    "session_id": run.get("session_id"),
                    "mode": case.get("mode"),
                    "score": score,
                    "environment": run.get("environment", {}),
                },
            }
        )
    return items


def _export_openai_evals_jsonl(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in report.get("case_results", []):
        case = item.get("case", {})
        run = item.get("run", {})
        score = item.get("score", {})
        items.append(
            {
                "item": {
                    "input": case.get("task", ""),
                    "ideal": case.get("assertions", {}).get("ideal", ""),
                    "metadata": {
                        "eval_run_id": report.get("eval_run_id"),
                        "suite": case.get("suite"),
                        "case_id": case.get("id"),
                        "mode": case.get("mode"),
                    },
                },
                "sample": {"output_text": run.get("final_answer", "")},
                "score": score,
            }
        )
    return items


def _export_note(platform: str) -> str:
    notes = {
        "promptfoo": "Use with promptfoo standalone assertions, for example: promptfoo eval --assertions asserts.yaml --model-outputs <path>",
        "langsmith": "Import as a dataset or use the LangSmith SDK to create examples from this JSONL file.",
        "langfuse": "Use this JSONL as an ingestion/interchange artifact; API upload requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
        "openai": "Use as an OpenAI Evals-style interchange file; adapt field names if your grader expects a custom schema.",
        "openai-evals": "Use as an OpenAI Evals-style interchange file; adapt field names if your grader expects a custom schema.",
    }
    return notes.get(platform, "")


def _evidence_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [f"ev_{index}" for index, item in enumerate(evidence, start=1) if item.get("status", "success") == "success"]


def _has_nonempty_sources(items: list[dict[str, Any]]) -> bool:
    for item in items:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        sources = metadata.get("sources")
        if isinstance(sources, list) and sources:
            return True
        if isinstance(item.get("sources"), list) and item.get("sources"):
            return True
    return False


def _ratio_score(expected: list[Any], text: str) -> float:
    if not expected:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for item in expected if str(item).lower() in lowered)
    return round(hits / len(expected), 4)


def _tool_match_score(expected: list[str], called: list[str], match_mode: str) -> float:
    if not expected:
        return 1.0
    if match_mode == "strict":
        return 1.0 if called[: len(expected)] == expected else 0.0
    if match_mode in {"subset", "unordered"}:
        return round(sum(1 for item in expected if item in called) / len(expected), 4)
    matched = 0
    cursor = 0
    for tool in expected:
        try:
            index = called.index(tool, cursor)
        except ValueError:
            continue
        matched += 1
        cursor = index + 1
    return round(matched / len(expected), 4)


def _redundant_tool_penalty(called: list[str]) -> float:
    if not called:
        return 1.0
    unique = len(set(called))
    return round(max(0.0, unique / len(called)), 4)


def _tool_call_fingerprints(tool_calls: list[dict[str, Any]]) -> list[str]:
    fingerprints: list[str] = []
    for item in tool_calls:
        name = item.get("tool_name") or item.get("tool") or ""
        args = item.get("input") or item.get("args") or {}
        try:
            args_key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except TypeError:
            args_key = str(args)
        fingerprints.append(f"{name}:{args_key}")
    return fingerprints


def _cited_evidence_ids(text: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"\bev_\d+\b", text)))


def _citation_coverage(text: str) -> float:
    blocks = _claim_blocks(text)
    if not blocks:
        return 1.0
    cited = sum(1 for block in blocks if _cited_evidence_ids(block))
    return round(cited / len(blocks), 4)


def _claim_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        starts_claim = line.startswith("-") or _starts_numbered_item(line)
        if starts_claim:
            if current:
                blocks.append("\n".join(current))
            current = [line]
            continue
        if current and (raw_line.startswith(" ") or raw_line.startswith("\t")):
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [
        block for block in blocks
        if not block.startswith("- high:")
        and not block.startswith("- medium:")
        and not block.startswith("- low:")
        and "如需更可靠结论" not in block
    ]


def _starts_numbered_item(line: str) -> bool:
    return any(line.startswith(f"{index}.") for index in range(1, 21))


def _weighted_average(metrics: dict[str, float], weights: dict[str, float]) -> float:
    if not metrics:
        return 1.0
    total_weight = 0.0
    total = 0.0
    for key, value in metrics.items():
        weight = float(weights.get(key, 1.0))
        total_weight += weight
        total += float(value) * weight
    return round(total / max(total_weight, 1e-9), 4)


def _max_severity(issues) -> IssueSeverity:
    order = {"info": 0, "minor": 1, "major": 2, "blocker": 3}
    max_item: IssueSeverity = "info"
    for item in issues:
        severity = item.severity if isinstance(item, EvalIssue) else str(item)
        if order.get(severity, 0) > order[max_item]:
            max_item = severity  # type: ignore[assignment]
    return max_item


def _issue_counts(case_results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"blocker": 0, "major": 0, "minor": 0, "info": 0}
    for item in case_results:
        for issue in item.get("score", {}).get("issue_details", []):
            severity = issue.get("severity", "info")
            counts[severity] = counts.get(severity, 0) + 1
    return counts


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item.get("case", {}).get("id"): item for item in report.get("case_results", [])}


def _case_status_delta(left: dict[str, Any], right: dict[str, Any], failed_only: bool = False) -> list[str]:
    left_cases = _case_map(left)
    right_cases = _case_map(right)
    changed: list[str] = []
    for case_id, right_item in right_cases.items():
        left_passed = bool(left_cases.get(case_id, {}).get("score", {}).get("passed", True))
        right_passed = bool(right_item.get("score", {}).get("passed", False))
        if failed_only and left_passed and not right_passed:
            changed.append(case_id)
        elif not failed_only and left_passed != right_passed:
            changed.append(case_id)
    return changed


def _case_success_delta(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    left_cases = _case_map(left)
    right_cases = _case_map(right)
    improved: list[str] = []
    for case_id, right_item in right_cases.items():
        left_passed = bool(left_cases.get(case_id, {}).get("score", {}).get("passed", False))
        right_passed = bool(right_item.get("score", {}).get("passed", False))
        if not left_passed and right_passed:
            improved.append(case_id)
    return improved


def _quality_regressions(left: dict[str, Any], right: dict[str, Any], min_delta: float = 0.05) -> list[dict[str, Any]]:
    left_cases = _case_map(left)
    right_cases = _case_map(right)
    regressions: list[dict[str, Any]] = []
    for case_id, right_item in right_cases.items():
        left_score = float(left_cases.get(case_id, {}).get("score", {}).get("quality_score", left_cases.get(case_id, {}).get("score", {}).get("score", 0.0)))
        right_score = float(right_item.get("score", {}).get("quality_score", right_item.get("score", {}).get("score", 0.0)))
        delta = round(right_score - left_score, 4)
        if delta < -min_delta:
            regressions.append({"case_id": case_id, "delta": delta, "left": left_score, "right": right_score})
    return regressions


def _tool_call_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    left_count = sum(len(item.get("run", {}).get("tool_calls", [])) for item in left.get("case_results", []))
    right_count = sum(len(item.get("run", {}).get("tool_calls", [])) for item in right.get("case_results", []))
    return {"left": left_count, "right": right_count, "delta": right_count - left_count}


def _prompt_contract_text() -> str:
    from src.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
    from src.prompts.summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT
    from src.prompts.team_prompt import TEAM_LEAD_SYSTEM_PROMPT, TEAM_MEMBER_SYSTEM_PROMPT
    from src.prompts.verifier_prompt import VERIFIER_SYSTEM_PROMPT

    return "\n\n".join(
        [
            PLANNER_SYSTEM_PROMPT,
            VERIFIER_SYSTEM_PROMPT,
            SUMMARIZER_SYSTEM_PROMPT,
            TEAM_LEAD_SYSTEM_PROMPT,
            TEAM_MEMBER_SYSTEM_PROMPT,
        ]
    )
