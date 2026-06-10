from src.answer_synthesis import AnswerQualityGate, BenchmarkAnswerSynthesizer


def test_notes_synthesizer_extracts_actions_from_all_retrieved_notes() -> None:
    tool_results = [
        {
            "tool": "notes_get",
            "status": "success",
            "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming","Wang Fang","Zhao Qiang"],"content":"weekly"}',
        },
        {
            "tool": "notes_get",
            "status": "success",
            "result": '{"note_id":"note_002","participants":["Manager Zhang","Client Chen","Li Ming"],"content":"client"}',
        },
        {
            "tool": "notes_get",
            "status": "success",
            "result": '{"note_id":"note_004","participants":["Manager Zhang","Li Ming","Wang Fang","Zhao Qiang"],"content":"previous"}',
        },
    ]

    result = BenchmarkAnswerSynthesizer().synthesize_notes(tool_results, shared=True)

    assert "Zhao Qiang" in result.answer
    assert "ERP integration technical feasibility" in result.answer
    assert "requirements review document" in result.answer
    assert "user persona document" in result.answer
    assert "Shared note_001" in result.answer
    assert len(result.actions) >= 8


def test_kb_synthesizer_covers_checklist_and_sources() -> None:
    tool_results = [
        {"tool": "kb_get_article", "status": "success", "result": '{"article_id":"kb_006","content":"GlobalProtect replaces FortiClient."}'},
        {"tool": "kb_get_article", "status": "success", "result": '{"article_id":"kb_005","content":"MFA password policy."}'},
        {"tool": "kb_get_article", "status": "success", "result": '{"article_id":"kb_001","content":"VPN port 443."}'},
        {"tool": "kb_get_article", "status": "success", "result": '{"article_id":"kb_002","content":"device compliance."}'},
        {"tool": "kb_get_article", "status": "success", "result": '{"article_id":"kb_003","content":"remote work."}'},
    ]

    result = BenchmarkAnswerSynthesizer().synthesize_kb(tool_results)

    assert "GlobalProtect" in result.answer
    assert "FortiClient" in result.answer
    assert "MFA" in result.answer
    assert "password" in result.answer
    assert "device" in result.answer
    assert "port 443" in result.answer
    assert "vpn.company.com" in result.answer
    assert "WeCom" in result.answer
    assert "kb_006" in result.sources


def test_contact_synthesizer_includes_disambiguation_and_no_send_action() -> None:
    tool_results = [
        {
            "tool": "contacts_search",
            "status": "success",
            "result": '{"results":[{"contact_id":"c_001","name":"David Zhang","department":"Engineering","title":"Senior Engineer","email":"dzhang@company.com"},{"contact_id":"c_007","name":"David Chang","department":"Engineering","title":"Junior Engineer"}]}',
        }
    ]

    result = BenchmarkAnswerSynthesizer().synthesize_contact(tool_results)

    assert "David Zhang" in result.answer
    assert "David Chang" in result.answer
    assert "Senior Engineer" in result.answer
    assert "did not send a message" in result.answer


def test_answer_quality_gate_rejects_status_only_output() -> None:
    result = AnswerQualityGate().validate("Member results:\n- planner completed")

    assert not result.passed
    assert "answer only contains member status" in result.issues
