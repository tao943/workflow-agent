from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.config import AppConfig
from src.evals import EvalCase, EvalHarness, _merge_scores, _run_result_payload


BenchmarkSource = Literal["local", "deepeval", "agentbench", "claw"]
BenchmarkMode = Literal["single", "team", "rag", "mcp", "skill", "tool_provider", "prompt_contract"]


@dataclass
class BenchmarkCase:
    id: str
    source: BenchmarkSource = "local"
    suite: str = "baseline_v1"
    official_task_id: str = ""
    protocol: str = "local"
    split: str = ""
    environment: str = ""
    fixtures: list[str] = field(default_factory=list)
    trials: int = 1
    grader: str = ""
    task: str = ""
    mode: BenchmarkMode = "single"
    agent: str | None = None
    team: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)
    rubric: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_category: str = ""
    unsupported_reason: str = ""
    assertions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkCase":
        if not payload.get("id"):
            raise ValueError("Benchmark case missing id")
        return cls(**payload)

    def to_eval_case(self) -> EvalCase:
        assertions = dict(self.assertions)
        if self.expected_artifacts:
            assertions["expected_artifacts"] = self.expected_artifacts
        if self.rubric:
            assertions["rubric"] = self.rubric
        return EvalCase(
            id=self.id,
            suite=self.suite,
            task=self.task,
            mode=self.mode,  # type: ignore[arg-type]
            agent=self.agent,
            team=self.team,
            offline=True,
            expected_tools=self.expected_tools,
            forbidden_tools=self.forbidden_tools,
            required_sections=self.required_sections,
            assertions=assertions,
            grading_policy=self.metrics.get("grading_policy", "weighted"),
            pass_threshold=float(self.metrics.get("pass_threshold", 0.75)),
            metric_weights=dict(self.metrics.get("metric_weights", {})),
            critical_assertions=list(self.metrics.get("critical_assertions", [])),
            tags=[self.source, self.failure_category] if self.failure_category else [self.source],
        )


@dataclass
class BenchmarkReport:
    benchmark_run_id: str
    source: str
    suite: str
    started_at: str
    duration_ms: int
    success_rate: float
    official_score: float
    local_score: float
    unsupported_count: int
    trial_results: list[dict[str, Any]]
    protocol_version: str
    dataset_revision: str
    environment_status: dict[str, Any]
    metric_summary: dict[str, float]
    failure_breakdown: dict[str, int]
    case_results: list[dict[str, Any]]


class BenchmarkHarness:
    def __init__(
        self,
        config: AppConfig,
        llm: Any | None = None,
        cases_dir: str | Path = "evals/benchmarks",
        imported_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.cases_dir = Path(cases_dir)
        self.output_dir = Path(output_dir or config.evals.benchmark_output_dir)
        self.imported_dir = Path(imported_dir or self.output_dir / "imported")
        self.reports_dir = self.output_dir / "reports"

    def list_suites(self) -> list[dict[str, Any]]:
        cases = self.load_cases()
        suites = sorted({case.suite for case in cases})
        return [
            {
                "suite": suite,
                "cases": len([case for case in cases if case.suite == suite]),
                "sources": sorted({case.source for case in cases if case.suite == suite}),
            }
            for suite in suites
        ]

    def load_cases(self, suite: str | None = None) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for root in [self.cases_dir, self.imported_dir]:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.jsonl")):
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if not line.strip() or line.strip().startswith("#"):
                        continue
                    try:
                        case = BenchmarkCase.from_dict(json.loads(line))
                    except Exception as exc:
                        raise ValueError(f"Invalid benchmark case {path}:{line_number}: {exc}") from exc
                    if suite is None or case.suite == suite:
                        cases.append(case)
        return cases

    def import_cases(self, source: str, path: str | Path) -> dict[str, Any]:
        if source not in {"agentbench", "claw"}:
            raise ValueError("source must be agentbench or claw")
        source_path = Path(path)
        if not source_path.exists():
            raise ValueError(f"Benchmark path not found: {path}")
        raw_items = _read_external_items(source_path)
        cases = [_external_item_to_case(source, item, index) for index, item in enumerate(raw_items, start=1)]
        self.imported_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.imported_dir / f"{source}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        output_path.write_text("\n".join(json.dumps(case.__dict__, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
        return {
            "source": source,
            "path": str(output_path),
            "imported": len(cases),
            "unsupported": sum(1 for case in cases if case.unsupported_reason),
        }

    def run_suite(self, suite: str, offline: bool = True, output_format: str = "default") -> BenchmarkReport:
        cases = self.load_cases(suite)
        if not cases:
            raise ValueError(f"Unknown benchmark suite or no cases: {suite}")
        benchmark_run_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        started = time.perf_counter()
        eval_harness = EvalHarness(self.config, llm=None if offline else self.llm, output_dir=self.output_dir / "eval_runs")
        case_results: list[dict[str, Any]] = []
        metric_totals: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        for case in cases:
            if case.unsupported_reason:
                case_results.append(
                    {
                        "case": case.__dict__,
                        "status": "unsupported",
                        "score": {"passed": False, "quality_score": 0.0, "issues": [case.unsupported_reason]},
                    }
                )
                continue
            eval_case = case.to_eval_case()
            result = eval_harness._run_case(benchmark_run_id, eval_case, offline=offline, output_format=output_format)
            scores = [evaluator.evaluate(result) for evaluator in eval_harness.evaluators]
            merged = _merge_scores(eval_case, scores)
            case_results.append(
                {
                    "case": case.__dict__,
                    "run": _run_result_payload(result),
                    "score": merged.__dict__,
                    "evaluator_scores": [score.__dict__ for score in scores],
                }
            )
            for key, value in merged.metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + float(value)
                metric_counts[key] = metric_counts.get(key, 0) + 1
        runnable = [item for item in case_results if item.get("status") != "unsupported"]
        passed = sum(1 for item in runnable if item.get("score", {}).get("passed"))
        metric_summary = {
            key: round(value / max(metric_counts.get(key, 1), 1), 4)
            for key, value in metric_totals.items()
        }
        report = BenchmarkReport(
            benchmark_run_id=benchmark_run_id,
            source="mixed",
            suite=suite,
            started_at=datetime.now().isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - started) * 1000),
            success_rate=round(passed / max(len(runnable), 1), 4),
            official_score=0.0,
            local_score=round(passed / max(len(runnable), 1), 4),
            unsupported_count=sum(1 for item in case_results if item.get("status") == "unsupported"),
            trial_results=[],
            protocol_version="local-benchmark.v1",
            dataset_revision="local",
            environment_status={"protocol": "local_adapter", "ok": True},
            metric_summary=metric_summary,
            failure_breakdown=_failure_breakdown(case_results),
            case_results=case_results,
        )
        self.save_report(report)
        return report

    def save_report(self, report: BenchmarkReport) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / f"{report.benchmark_run_id}.json"
        path.write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_report(self, benchmark_run_id: str) -> dict[str, Any] | None:
        path = self.reports_dir / f"{benchmark_run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _read_external_items(path: Path) -> list[dict[str, Any]]:
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    items: list[dict[str, Any]] = []
    for item_path in paths:
        text = item_path.read_text(encoding="utf-8")
        if item_path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, list):
                items.extend(item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                candidates = payload.get("tasks") or payload.get("cases") or payload.get("data") or [payload]
                items.extend(item for item in candidates if isinstance(item, dict))
            continue
        for line in text.splitlines():
            if line.strip():
                items.append(json.loads(line))
    return items


def _external_item_to_case(source: str, item: dict[str, Any], index: int) -> BenchmarkCase:
    task = str(item.get("task") or item.get("instruction") or item.get("prompt") or item.get("question") or "")
    case_id = str(item.get("id") or item.get("task_id") or item.get("name") or f"{source}_{index}")
    environment = str(item.get("environment") or item.get("env") or item.get("category") or "")
    fixtures = item.get("fixture") or item.get("fixtures") or []
    if isinstance(fixtures, str):
        fixtures = [fixtures]
    unsupported = ""
    if not task:
        unsupported = "missing task/instruction field"
    elif any(token in environment.lower() for token in ["browser", "gui", "terminal", "shell", "webshop", "os"]):
        unsupported = f"unsupported_environment:{environment}"
    return BenchmarkCase(
        id=case_id,
        source=source,  # type: ignore[arg-type]
        suite=f"{source}_imported",
        official_task_id=case_id,
        protocol=source,
        split=str(item.get("split") or ""),
        environment=environment,
        fixtures=list(fixtures) if isinstance(fixtures, list) else [],
        trials=int(item.get("trials") or 1),
        grader=str(item.get("grader") or item.get("grade") or ""),
        task=task,
        mode="single",
        agent="build",
        rubric=str(item.get("rubric") or item.get("expected") or item.get("answer") or ""),
        metrics={"grading_policy": "weighted", "pass_threshold": 0.75},
        unsupported_reason=unsupported,
        assertions={"contains": item.get("contains", []) if isinstance(item.get("contains"), list) else []},
    )


def _failure_breakdown(case_results: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for item in case_results:
        if item.get("status") == "unsupported":
            breakdown["unsupported_environment"] = breakdown.get("unsupported_environment", 0) + 1
            continue
        if item.get("score", {}).get("passed"):
            continue
        case = item.get("case", {})
        category = case.get("failure_category") or "unknown"
        breakdown[category] = breakdown.get(category, 0) + 1
    return breakdown
