import json
import subprocess
import sys
from pathlib import Path

from outputs.swebench.tuning_audit import (
    assemble_case_record,
    build_case_metrics,
    canonical_json,
    load_json_evidence,
    render_markdown,
    trace_evidence_from_snapshot,
)


def test_build_case_metrics_keeps_three_official_metrics_separate():
    report = {
        "resolved": False,
        "tests_status": {
            "FAIL_TO_PASS": {"success": ["fix-a", "fix-b"], "failure": []},
            "PASS_TO_PASS": {"success": ["reg-a", "reg-b", "reg-c"], "failure": ["reg-d"]},
        },
    }

    result = build_case_metrics(report)

    assert result["official_resolved"] is False
    assert result["fail_to_pass"] == {"passed": 2, "failed": 0, "rate": 1.0}
    assert result["pass_to_pass"] == {"passed": 3, "failed": 1, "rate": 0.75}
    assert result["overall"] == {"passed": 5, "failed": 1, "total": 6, "rate": 5 / 6, "over_90_percent": False}


def test_missing_or_corrupt_json_is_reported_as_missing_evidence(tmp_path):
    missing = load_json_evidence(tmp_path / "missing.json")
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{broken", encoding="utf-8")
    broken = load_json_evidence(broken_path)

    assert missing["evidence_level"] == "missing"
    assert missing["error"] == "file_not_found"
    assert broken["evidence_level"] == "missing"
    assert broken["error"] == "invalid_json"


def test_trace_snapshot_is_offline_authority_and_preserves_operations():
    snapshot = {
        "trace-1": {
            "http_status": 200,
            "operations": ["a2a.remote", "ChatOpenAI"],
            "queried_at": "2026-08-29T15:50:00+08:00",
        }
    }

    result = trace_evidence_from_snapshot("trace-1", snapshot, "outputs/swebench/otlp-jaeger-snapshot.json")

    assert result["verified"] is True
    assert result["operations"] == ["a2a.remote", "ChatOpenAI"]
    assert result["evidence_level"] == "direct"
    assert result["evidence_paths"] == ["outputs/swebench/otlp-jaeger-snapshot.json"]


def test_markdown_and_json_are_deterministic_and_disclose_reconstruction():
    audit = {
        "schema_version": 1,
        "title": "A2A 调优审计",
        "executive_summary": {"all_five_over_90_percent": True},
        "cases": [],
        "timeline": [{"stage": "基线", "evidence_level": "direct", "summary": "0/5 resolved"}],
        "reconstructed_findings": [{"finding": "代理影响 localhost", "evidence_level": "reconstructed"}],
        "evidence_gaps": ["早期 attempt 响应被覆盖"],
    }

    first = canonical_json(audit)
    second = canonical_json(audit)
    markdown = render_markdown(audit)

    assert first == second
    assert json.loads(first)["schema_version"] == 1
    assert "```mermaid" in markdown
    assert "历史重建" in markdown
    assert "早期 attempt 响应被覆盖" in markdown


def test_assemble_case_record_links_patch_report_and_trace_provenance():
    report = {
        "resolved": True,
        "tests_status": {
            "FAIL_TO_PASS": {"success": ["fixed"], "failure": []},
            "PASS_TO_PASS": {"success": ["kept"], "failure": []},
        },
    }
    snapshot = {"trace-x": {"http_status": 200, "operations": ["a2a.remote", "ChatOpenAI"], "span_count": 2}}

    result = assemble_case_record(
        instance_id="case-x",
        final_report=report,
        v1_report=None,
        final_patch="diff --git a/a b/a\n",
        baseline_status="unresolved",
        trace_id="trace-x",
        trace_snapshot=snapshot,
        evidence_paths=["final-report.json", "predictions.jsonl"],
        trace_snapshot_path="trace-snapshot.json",
    )

    assert result["final_metrics"]["overall"]["rate"] == 1.0
    assert result["final_patch"]["sha256"] == "9e3e63fac9c92100f0f15e616e1abf395028edc8912bebfc42044c65eb17e114"
    assert result["trace"]["verified"] is True
    assert result["evidence_paths"] == ["final-report.json", "predictions.jsonl"]


def test_audit_cli_builds_json_and_markdown_offline(tmp_path):
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "outputs" / "swebench" / "build_tuning_audit.py"), "--root", str(root), "--output-dir", str(tmp_path), "--offline"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "tuning-audit.json").read_text(encoding="utf-8"))
    report = (tmp_path / "tuning-audit-report.md").read_text(encoding="utf-8")
    assert len(audit["cases"]) == 5
    assert audit["executive_summary"]["all_five_over_90_percent"] is True
    assert "失败驱动调优矩阵" in report
