from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.config import AppConfig


ProtocolName = Literal["agentbench", "claw", "deepeval"]


@dataclass
class ProtocolDoctorResult:
    protocol: ProtocolName
    ok: bool
    checks: dict[str, dict[str, Any]]
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ProtocolTrialResult:
    trial: int
    status: str
    score: float = 0.0
    passed: bool = False
    output_path: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OfficialBenchmarkReport:
    benchmark_run_id: str
    protocol: ProtocolName
    protocol_version: str
    dataset_revision: str
    split: str
    suite: str
    started_at: str
    duration_ms: int
    official_score: float
    local_score: float
    unsupported_count: int
    trial_results: list[dict[str, Any]]
    environment_status: dict[str, Any]
    status: str
    notes: list[str] = field(default_factory=list)


class BenchmarkProtocolRunner:
    protocol: ProtocolName

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.output_dir = Path(config.benchmarks.protocol_output_dir)

    def doctor(self) -> ProtocolDoctorResult:
        raise NotImplementedError

    def run(self, split: str, suite: str, limit: int | None = None, trials: int | None = None) -> OfficialBenchmarkReport:
        started = time.perf_counter()
        doctor = self.doctor()
        trial_count = max(1, int(trials or self.config.benchmarks.default_trials))
        run_id = f"{self.protocol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        if not doctor.ok:
            report = OfficialBenchmarkReport(
                benchmark_run_id=run_id,
                protocol=self.protocol,
                protocol_version=self._protocol_version(),
                dataset_revision=self._dataset_revision(),
                split=split,
                suite=suite,
                started_at=datetime.now().isoformat(timespec="seconds"),
                duration_ms=int((time.perf_counter() - started) * 1000),
                official_score=0.0,
                local_score=0.0,
                unsupported_count=limit or 0,
                trial_results=[],
                environment_status=doctor.__dict__,
                status="environment_not_ready",
                notes=doctor.recommendations,
            )
            self.save_report(report)
            return report
        trials_payload = self._run_trials(split=split, suite=suite, limit=limit, trials=trial_count)
        passed = sum(1 for item in trials_payload if item.passed)
        pass_at_all_trials = 1.0 if trials_payload and passed == len(trials_payload) else 0.0
        report = OfficialBenchmarkReport(
            benchmark_run_id=run_id,
            protocol=self.protocol,
            protocol_version=self._protocol_version(),
            dataset_revision=self._dataset_revision(),
            split=split,
            suite=suite,
            started_at=datetime.now().isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - started) * 1000),
            official_score=pass_at_all_trials,
            local_score=round(sum(item.score for item in trials_payload) / max(len(trials_payload), 1), 4),
            unsupported_count=0,
            trial_results=[item.__dict__ for item in trials_payload],
            environment_status=doctor.__dict__,
            status="completed",
            notes=self._notes(),
        )
        self.save_report(report)
        return report

    def export_report(self, run_id: str) -> dict[str, Any] | None:
        report = self.load_report(run_id)
        if not report:
            return None
        export_dir = self.output_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / f"{run_id}.official.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": str(path), "items": len(report.get("trial_results", [])), "protocol": report.get("protocol")}

    def load_report(self, run_id: str) -> dict[str, Any] | None:
        path = self.output_dir / "reports" / f"{run_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_report(self, report: OfficialBenchmarkReport) -> Path:
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{report.benchmark_run_id}.json"
        path.write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _run_trials(self, split: str, suite: str, limit: int | None, trials: int) -> list[ProtocolTrialResult]:
        return [
            ProtocolTrialResult(
                trial=index,
                status="protocol_ready_manual_execution_required",
                score=0.0,
                passed=False,
                metadata={"split": split, "suite": suite, "limit": limit},
            )
            for index in range(1, trials + 1)
        ]

    def _protocol_version(self) -> str:
        return "protocol-runner.v1"

    def _dataset_revision(self) -> str:
        return "unknown"

    def _notes(self) -> list[str]:
        return ["Protocol runner checked the official environment shape; leaderboard parity requires the benchmark's own CLI/grader output."]


class DeepEvalProtocolRunner(BenchmarkProtocolRunner):
    protocol: ProtocolName = "deepeval"

    def doctor(self) -> ProtocolDoctorResult:
        checks = {
            "deepeval_installed": _python_import_check("deepeval"),
            "evaluation_model": {"ok": bool(self.config.evals.evaluation_model), "value": self.config.evals.evaluation_model},
            "mode": {"ok": self.config.evals.deepeval_mode in {"local", "native"}, "value": self.config.evals.deepeval_mode},
        }
        ok = all(item.get("ok") for item in checks.values())
        recommendations = []
        if not checks["deepeval_installed"]["ok"]:
            recommendations.append("Install deepeval to enable native tracing metrics.")
        if self.config.evals.deepeval_mode != "native":
            recommendations.append("Use --deepeval-mode native for official DeepEval tracing mode.")
        return ProtocolDoctorResult("deepeval", ok, checks, recommendations)


class AgentBenchProtocolRunner(BenchmarkProtocolRunner):
    protocol: ProtocolName = "agentbench"

    def doctor(self) -> ProtocolDoctorResult:
        root = Path(self.config.benchmarks.agentbench_path) if self.config.benchmarks.agentbench_path else None
        checks = {
            "checkout_path": _path_check(root, ["src", "configs", "data"]),
            "docker": _command_check("docker", ["docker", "--version"]),
            "python": {"ok": True, "value": platform.python_version()},
            "redis_hint": {"ok": bool(shutil.which("redis-server") or os.getenv("REDIS_URL")), "value": os.getenv("REDIS_URL", "")},
            "resource_heavy_allowed": {"ok": bool(self.config.benchmarks.allow_resource_heavy), "value": self.config.benchmarks.allow_resource_heavy},
        }
        required = ["checkout_path", "docker", "python"]
        ok = all(checks[key].get("ok") for key in required)
        recommendations = []
        if not checks["checkout_path"]["ok"]:
            recommendations.append("Set benchmarks.agentbench_path to a local THUDM/AgentBench checkout.")
        if not checks["docker"]["ok"]:
            recommendations.append("Install/start Docker before running AgentBench task workers.")
        if not checks["redis_hint"]["ok"]:
            recommendations.append("AgentBench often needs Redis/task workers; configure REDIS_URL or start Redis if your selected suite requires it.")
        return ProtocolDoctorResult("agentbench", ok, checks, recommendations)

    def _run_trials(self, split: str, suite: str, limit: int | None, trials: int) -> list[ProtocolTrialResult]:
        heavy = suite.lower() in {"webshop", "mind2web"}
        if heavy and not self.config.benchmarks.allow_resource_heavy:
            return [
                ProtocolTrialResult(
                    trial=1,
                    status="unsupported_or_deferred",
                    error=f"{suite} is resource-heavy and benchmarks.allow_resource_heavy=false",
                    metadata={"split": split, "suite": suite, "limit": limit},
                )
            ]
        wrapper = self._write_agentbench_wrapper_config(split, suite, limit)
        return [
            ProtocolTrialResult(
                trial=index,
                status="protocol_ready_manual_assigner_required",
                score=0.0,
                passed=False,
                output_path=wrapper,
                metadata={"split": split, "suite": suite, "limit": limit, "wrapper_config": wrapper},
            )
            for index in range(1, trials + 1)
        ]

    def _write_agentbench_wrapper_config(self, split: str, suite: str, limit: int | None) -> str:
        path = self.output_dir / "agentbench" / f"wrapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "agentbench",
            "split": split,
            "suite": suite,
            "limit": limit,
            "agent_wrapper": "python -m src.main <task> --agent build --yes",
            "note": "Use this config as the local project wrapper when invoking AgentBench assigner/task server.",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


class ClawEvalProtocolRunner(BenchmarkProtocolRunner):
    protocol: ProtocolName = "claw"

    def doctor(self) -> ProtocolDoctorResult:
        root = Path(self.config.benchmarks.claw_eval_path) if self.config.benchmarks.claw_eval_path else None
        checks = {
            "checkout_path": _path_check(root, ["scripts", "tasks", "src"]),
            "claw_eval_cli": _python_module_check("claw_eval.cli", ["--help"], timeout=8),
            "api_key": {"ok": bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")), "value": "configured" if (os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")) else ""},
            "sandbox": _command_check("bash", ["bash", "--version"], timeout=5),
            "docker_daemon": _command_check("docker", ["docker", "info", "--format", "{{.ServerVersion}}"], timeout=20),
            "sandbox_image": _command_check("docker", ["docker", "image", "inspect", "claw-eval-agent:latest", "--format", "{{.Id}}"], timeout=20),
        }
        ok = bool(checks["checkout_path"]["ok"] and checks["api_key"]["ok"])
        recommendations = []
        if not checks["checkout_path"]["ok"]:
            recommendations.append("Set benchmarks.claw_eval_path to a local claw-eval checkout with fixtures.")
        if not checks["api_key"]["ok"]:
            recommendations.append("Configure the judge/model API key required by Claw-Eval before running official trials.")
        if not checks["claw_eval_cli"]["ok"]:
            recommendations.append("Install the claw-eval CLI in the current environment.")
        if not checks["sandbox"]["ok"]:
            recommendations.append("Install/enable bash or WSL if you want to run Claw-Eval's shell sandbox helper scripts on Windows.")
        if not checks["docker_daemon"]["ok"]:
            recommendations.append("Start Docker Desktop before running Claw-Eval sandbox tasks.")
        if not checks["sandbox_image"]["ok"]:
            recommendations.append("Build the sandbox image with scripts/claw_eval.ps1 build-image --config config_workflow_agent.yaml --context . --dockerfile Dockerfile.agent.")
        return ProtocolDoctorResult("claw", ok, checks, recommendations)

    def _run_trials(self, split: str, suite: str, limit: int | None, trials: int) -> list[ProtocolTrialResult]:
        if split == "multimodal" and not self.config.benchmarks.allow_resource_heavy:
            return [
                ProtocolTrialResult(
                    trial=1,
                    status="unsupported_or_deferred",
                    error="multimodal/video fixtures are deferred unless benchmarks.allow_resource_heavy=true",
                    metadata={"split": split, "suite": suite, "limit": limit},
                )
            ]
        run_config = self._write_claw_run_manifest(split, suite, limit, trials)
        return [
            ProtocolTrialResult(
                trial=index,
                status="protocol_ready_manual_sandbox_required",
                score=0.0,
                passed=False,
                output_path=run_config,
                metadata={"split": split, "suite": suite, "limit": limit, "pass_k_requires_all_trials": True},
            )
            for index in range(1, trials + 1)
        ]

    def _write_claw_run_manifest(self, split: str, suite: str, limit: int | None, trials: int) -> str:
        path = self.output_dir / "claw" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "claw",
            "split": split,
            "suite": suite,
            "limit": limit,
            "trials": trials,
            "suggested_command": f"powershell -ExecutionPolicy Bypass -File scripts/claw_eval.ps1 batch --config config_workflow_agent.yaml --sandbox --trials {trials} --parallel 1 --filter {suite}",
            "windows_env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            "note": "Official leaderboard parity requires Claw-Eval's own sandbox, fixtures, grader, and three successful trajectories.",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def make_protocol_runner(protocol: str, config: AppConfig) -> BenchmarkProtocolRunner:
    if protocol == "agentbench":
        return AgentBenchProtocolRunner(config)
    if protocol == "claw":
        return ClawEvalProtocolRunner(config)
    if protocol == "deepeval":
        return DeepEvalProtocolRunner(config)
    raise ValueError("protocol must be agentbench, claw, or deepeval")


def _python_import_check(module: str) -> dict[str, Any]:
    try:
        __import__(module)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def _path_check(root: Path | None, expected_children: list[str]) -> dict[str, Any]:
    if not root:
        return {"ok": False, "error": "path_not_configured"}
    if not root.exists():
        return {"ok": False, "path": str(root), "error": "path_not_found"}
    missing = [child for child in expected_children if not (root / child).exists()]
    return {"ok": not missing, "path": str(root), "missing": missing}


def _command_check(name: str, command: list[str], timeout: int = 5) -> dict[str, Any]:
    if not shutil.which(name):
        return {"ok": False, "error": "command_not_found", "command": command}
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": command}
    output = (result.stdout or result.stderr or "").strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "output": output[:500],
    }


def _python_module_check(module: str, args: list[str] | None = None, timeout: int = 5) -> dict[str, Any]:
    command = [sys.executable, "-m", module, *(args or [])]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": command}
    output = (result.stdout or result.stderr or "").strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "output": output[:500],
    }
