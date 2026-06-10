from src.evidence import verify_final_answer_evidence
from src.synthesis import LeadSynthesizer, Recommendation, SynthesisClaim, validate_final_answer_contract


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _GoodSynthesisLLM:
    def invoke(self, messages):
        return _FakeResponse(
            "\n".join(
                [
                    "## 当前项目结构分析",
                    "- LLM_SELECTED: CLI 入口集中在 src/main.py。证据：ev_2",
                    "",
                    "## 主要问题",
                    "- TeamRuntime 和 synthesis 边界需要继续收敛。证据：ev_3",
                    "",
                    "## 优化建议",
                    "- 拆分 CLI command handlers，并保留现有参数兼容性。证据：ev_2",
                    "- 将最终答案生成保持在 LeadSynthesizer 内。证据：ev_3",
                    "",
                    "## 优先级路线",
                    "- high: 先拆 CLI command handlers。证据：ev_2",
                    "",
                    "## 风险与下一步",
                    "- 拆分时需要用测试保护原 CLI 行为。证据：ev_2",
                ]
            )
        )


class _BadSynthesisLLM:
    def invoke(self, messages):
        return _FakeResponse("Member results:\n- planner completed\n- researcher completed")


class _ErrorSynthesisLLM:
    def invoke(self, messages):
        raise RuntimeError("boom")


def _member_results(task_metadata=None):
    return [
        {
            "member": "researcher",
            "role": "researcher",
            "status": "completed",
            "task_metadata": task_metadata or {},
            "evidence": [
                {"tool": "list_files", "status": "success", "args": {"path": "."}, "metadata": {"count": 42}, "output": "src/main.py\nsrc/team.py\n"},
                {"tool": "read_file", "status": "success", "args": {"path": "src/main.py"}, "metadata": {"path": "D:/repo/src/main.py"}, "output": "argparse runtime inspect"},
                {"tool": "read_file", "status": "success", "args": {"path": "src/team.py"}, "metadata": {"path": "D:/repo/src/team.py"}, "output": "class TeamRuntime"},
            ],
        }
    ]


def test_lead_synthesizer_generates_cli_and_team_recommendations():
    result = LeadSynthesizer().synthesize("分析当前项目结构并给出优化建议", "default-dev-team", _member_results())
    titles = {item.title for item in result.recommendations}

    assert "拆分 CLI command handlers" in titles
    assert "继续拆分 TeamRuntime 边界" in titles
    assert all(item.evidence_ids for item in result.recommendations)


def test_lead_synthesizer_renders_user_facing_answer():
    synthesizer = LeadSynthesizer()
    result = synthesizer.synthesize("分析当前项目结构并给出优化建议", "default-dev-team", _member_results())
    answer = synthesizer.render_final_answer("分析当前项目结构并给出优化建议", "default-dev-team", result, _member_results())

    assert "## 当前项目结构分析" in answer
    assert "## 主要问题" in answer
    assert "## 优化建议" in answer
    assert "## 优先级路线" in answer
    assert "证据：ev_" in answer
    assert "Member results" not in answer


def test_lead_synthesizer_uses_testing_template_from_metadata():
    metadata = {"analysis_strategy": {"analysis_subtype": "testing", "depth": "standard"}}
    synthesizer = LeadSynthesizer()
    result = synthesizer.synthesize("分析当前测试体系还有哪些缺口", "default-dev-team", _member_results(metadata))
    answer = synthesizer.render_final_answer("分析当前测试体系还有哪些缺口", "default-dev-team", result, _member_results(metadata))

    assert result.analysis_subtype == "testing"
    assert "## 测试体系现状" in answer
    assert "## 新增测试建议" in answer


def test_lead_synthesizer_uses_benchmark_template_from_task():
    synthesizer = LeadSynthesizer()
    result = synthesizer.synthesize("根据 Claw-Eval benchmark 结果分析失败原因", "default-dev-team", _member_results())
    answer = synthesizer.render_final_answer("根据 Claw-Eval benchmark 结果分析失败原因", "default-dev-team", result, _member_results())

    assert result.analysis_subtype == "benchmark"
    assert "## Benchmark 现状" in answer
    assert "## 优化队列" in answer


def test_synthesis_contract_rejects_missing_evidence_id():
    issues = validate_final_answer_contract(
        [SynthesisClaim("claim", ["ev_missing"])],
        [
            Recommendation(
                title="rec",
                problem="problem",
                suggestion="suggestion",
                priority="high",
                evidence_ids=["ev_missing"],
                risk="risk",
                next_action="next",
            )
        ],
        [],
    )

    assert any("missing evidence id" in item for item in issues)


def test_final_answer_gate_rejects_status_only_output():
    check = verify_final_answer_evidence("Member results:\n- planner completed\n- researcher completed", {"ev_1"})

    assert not check.passed
    assert any("missing final answer section" in item for item in check.issues)


def test_llm_synthesis_candidate_is_used_after_evidence_gate_passes():
    synthesizer = LeadSynthesizer()
    result = synthesizer.synthesize("分析当前项目结构并给出优化建议", "default-dev-team", _member_results())

    answer, notes = synthesizer.render_with_optional_llm("分析当前项目结构并给出优化建议", "default-dev-team", result, _member_results(), _GoodSynthesisLLM())

    assert "LLM_SELECTED" in answer
    assert notes == ["llm_synthesis.passed"]


def test_llm_synthesis_falls_back_when_candidate_fails_evidence_gate():
    synthesizer = LeadSynthesizer()
    result = synthesizer.synthesize("分析当前项目结构并给出优化建议", "default-dev-team", _member_results())

    answer, notes = synthesizer.render_with_optional_llm("分析当前项目结构并给出优化建议", "default-dev-team", result, _member_results(), _BadSynthesisLLM())

    assert "LLM_SELECTED" not in answer
    assert "Member results" not in answer
    assert "llm_synthesis.fallback" in notes


def test_llm_synthesis_falls_back_when_model_errors():
    synthesizer = LeadSynthesizer()
    result = synthesizer.synthesize("分析当前项目结构并给出优化建议", "default-dev-team", _member_results())

    answer, notes = synthesizer.render_with_optional_llm("分析当前项目结构并给出优化建议", "default-dev-team", result, _member_results(), _ErrorSynthesisLLM())

    assert "LLM_SELECTED" not in answer
    assert any(item.startswith("llm_synthesis.error:") for item in notes)
