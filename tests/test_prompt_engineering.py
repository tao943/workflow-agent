import json

from src.evidence import verify_task_evidence
from src.nodes.planner import _fallback_plan, _normalize_steps
from src.nodes.summarizer import _fallback_summary
from src.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT
from src.prompts.summarizer_prompt import SUMMARIZER_SYSTEM_PROMPT
from src.prompts.team_prompt import TEAM_LEAD_SYSTEM_PROMPT, TEAM_MEMBER_SYSTEM_PROMPT
from src.prompts.templates import PROMPT_VERSION
from src.team import TeamRuntime


def test_prompts_are_versioned_and_schema_driven():
    assert PROMPT_VERSION in PLANNER_SYSTEM_PROMPT
    assert "Output Schema" in PLANNER_SYSTEM_PROMPT
    assert "evidence-first" in TEAM_LEAD_SYSTEM_PROMPT
    assert "evidence_ids" in TEAM_MEMBER_SYSTEM_PROMPT
    assert "已证实结论" in SUMMARIZER_SYSTEM_PROMPT


def test_planner_normalizes_required_evidence_from_json():
    payload = json.loads(
        """
        {
          "steps": [
            {
              "description": "List project files",
              "tool_name": "list_files",
              "tool_args": {"path": "."},
              "required_evidence": ["project file list"],
              "done_criteria": "file list exists"
            }
          ]
        }
        """
    )

    steps = _normalize_steps(payload["steps"], {"list_files"}, "分析项目结构")

    assert steps[0]["tool_name"] == "list_files"
    assert steps[0]["tool_args"] == {"path": "."}
    assert steps[0]["required_evidence"] == ["project file list"]


def test_project_analysis_fallback_requires_file_tools():
    steps = _fallback_plan("分析当前项目结构并给出优化建议", {"list_files", "read_file", "search_files"})
    tools = {step["tool_name"] for step in steps}

    assert {"list_files", "read_file", "search_files"}.issubset(tools)


def test_evidence_cards_do_not_include_full_raw_output():
    runtime = TeamRuntime.__new__(TeamRuntime)
    evidence = [
        {
            "tool": "read_file",
            "status": "success",
            "args": {"path": "src/main.py"},
            "output": "x" * 5000,
            "metadata": {"path": "src/main.py"},
            "raw_output_path": "outputs/artifacts/read_file.txt",
        }
    ]

    text = TeamRuntime._with_evidence_summary(runtime, "Task", evidence)

    assert "Evidence Cards:" in text
    assert "outputs/artifacts/read_file.txt" in text
    assert "x" * 1000 not in text


def test_reviewer_detects_missing_required_tools():
    check = verify_task_evidence(
        {"required_tools": ["list_files"], "required_evidence": ["project file list"]},
        {"final_answer": "已扫描项目结构。"},
        [],
    )

    assert not check.passed
    assert "required tool was not called: list_files" in check.issues


def test_fallback_summary_uses_evidence_sections():
    summary = _fallback_summary(
        {
            "user_task": "测试",
            "plan": [],
            "execution_log": [],
            "tool_results": [],
            "error": None,
        }
    )

    assert "## 已证实结论" in summary
    assert "## 合理推断" in summary
    assert "## 未验证假设" in summary
