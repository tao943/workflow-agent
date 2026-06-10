import json
from pathlib import Path
from uuid import uuid4

from src.benchmark_protocols import BenchmarkProtocolRunner, ProtocolDoctorResult, ProtocolTrialResult, make_protocol_runner
from src.benchmarks import BenchmarkCase, BenchmarkHarness
from src.config import AppConfig


def _benchmark_output_dir(name: str) -> Path:
    path = Path("outputs") / "test_benchmarks" / f"{name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_benchmark_case_converts_to_eval_case() -> None:
    case = BenchmarkCase(
        id="bench_case",
        source="local",
        official_task_id="official_1",
        protocol="agentbench",
        split="dev",
        environment="dbbench",
        fixtures=["fixture.json"],
        trials=3,
        grader="official",
        task="calculate 1 + 1",
        mode="single",
        agent="build",
        expected_tools=["calculator"],
        rubric="Must calculate correctly.",
        metrics={"grading_policy": "weighted", "pass_threshold": 0.8},
    )

    eval_case = case.to_eval_case()

    assert eval_case.id == "bench_case"
    assert eval_case.expected_tools == ["calculator"]
    assert eval_case.assertions["rubric"] == "Must calculate correctly."
    assert eval_case.pass_threshold == 0.8
    assert case.official_task_id == "official_1"
    assert case.fixtures == ["fixture.json"]


def test_benchmark_harness_lists_baseline_suite() -> None:
    harness = BenchmarkHarness(AppConfig(), output_dir=_benchmark_output_dir("list"))

    suites = {item["suite"]: item["cases"] for item in harness.list_suites()}

    assert suites["baseline_v1"] == 30


def test_benchmark_import_marks_unsupported_environment() -> None:
    output_dir = _benchmark_output_dir("import")
    source_path = output_dir / "agentbench_sample.jsonl"
    source_path.write_text(
        json.dumps({"id": "browser_task", "environment": "browser", "instruction": "Open a web page"}, ensure_ascii=False) + "\n"
        + json.dumps({"id": "cli_task", "instruction": "Summarize this task"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    harness = BenchmarkHarness(AppConfig(), output_dir=output_dir)

    imported = harness.import_cases("agentbench", source_path)
    cases = harness.load_cases("agentbench_imported")

    assert imported["imported"] == 2
    assert imported["unsupported"] == 1
    assert any(case.unsupported_reason.startswith("unsupported_environment") for case in cases)
    assert all(case.protocol == "agentbench" for case in cases)


def test_benchmark_run_saves_report_for_small_suite() -> None:
    output_dir = _benchmark_output_dir("run")
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True)
    cases_dir.joinpath("mini.jsonl").write_text(
        json.dumps(
            {
                "id": "mini_tool_provider",
                "source": "local",
                "suite": "mini",
                "task": "list tools",
                "mode": "tool_provider",
                "expected_tools": ["calculator"],
                "metrics": {"grading_policy": "weighted", "pass_threshold": 0.5},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    harness = BenchmarkHarness(AppConfig(), cases_dir=cases_dir, output_dir=output_dir)

    report = harness.run_suite("mini", offline=True, output_format="json")
    loaded = harness.load_report(report.benchmark_run_id)

    assert report.success_rate == 1.0
    assert report.protocol_version == "local-benchmark.v1"
    assert report.official_score == 0.0
    assert report.unsupported_count == 0
    assert loaded is not None
    assert loaded["benchmark_run_id"] == report.benchmark_run_id


def test_protocol_doctor_reports_missing_agentbench_checkout() -> None:
    config = AppConfig()
    config.benchmarks.agentbench_path = ""

    result = make_protocol_runner("agentbench", config).doctor()

    assert not result.ok
    assert result.checks["checkout_path"]["error"] == "path_not_configured"


def test_official_runner_passes_only_when_all_trials_pass() -> None:
    class PassingRunner(BenchmarkProtocolRunner):
        protocol = "claw"

        def doctor(self) -> ProtocolDoctorResult:
            return ProtocolDoctorResult("claw", True, {"fixture": {"ok": True}})

        def _run_trials(self, split: str, suite: str, limit: int | None, trials: int) -> list[ProtocolTrialResult]:
            return [
                ProtocolTrialResult(trial=1, status="completed", score=1.0, passed=True),
                ProtocolTrialResult(trial=2, status="completed", score=1.0, passed=True),
                ProtocolTrialResult(trial=3, status="completed", score=0.0, passed=False),
            ]

    runner = PassingRunner(AppConfig())
    runner.output_dir = _benchmark_output_dir("official")

    report = runner.run(split="general", suite="general", trials=3)

    assert report.official_score == 0.0
    assert report.local_score == 0.6667
    assert len(report.trial_results) == 3
