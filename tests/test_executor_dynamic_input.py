from src.nodes.executor import _normalize_dynamic_tool_input


def test_notes_share_placeholder_recipients_are_repaired_from_note_get() -> None:
    tool_results = [
        {
            "tool": "notes_get",
            "status": "success",
            "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}',
        }
    ]

    normalized = _normalize_dynamic_tool_input(
        "notes_share",
        {"note_id": "note_001", "recipients": ["participants from extracted attendees"]},
        tool_results,
    )

    assert normalized["recipients"] == ["Manager Zhang", "Li Ming"]


def test_notes_share_placeholder_recipients_fallback_to_notes_list() -> None:
    tool_results = [
        {
            "tool": "notes_list",
            "status": "success",
            "result": '{"notes":[{"note_id":"note_001","participants":["Manager Zhang","Li Ming","Wang Fang"]}]}',
        }
    ]

    normalized = _normalize_dynamic_tool_input(
        "notes_share",
        {"note_id": "note_001", "recipients": ["attendees_from_note_001"]},
        tool_results,
    )

    assert normalized["recipients"] == ["Manager Zhang", "Li Ming", "Wang Fang"]


def test_notes_share_placeholder_note_id_is_repaired_from_notes_list() -> None:
    tool_results = [
        {
            "tool": "notes_list",
            "status": "success",
            "result": '{"notes":[{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}]}',
        }
    ]

    normalized = _normalize_dynamic_tool_input(
        "notes_share",
        {"note_id": "<Feb 23周会笔记ID>", "recipients": ["attendees_from_note_001"]},
        tool_results,
    )

    assert normalized["note_id"] == "note_001"
    assert normalized["recipients"] == ["Manager Zhang", "Li Ming"]


def test_notes_get_placeholder_note_id_uses_description_and_notes_list() -> None:
    tool_results = [
        {
            "tool": "notes_list",
            "status": "success",
            "result": '{"notes":[{"note_id":"note_001"},{"note_id":"note_002"},{"note_id":"note_004"}]}',
        },
        {
            "tool": "notes_get",
            "status": "success",
            "result": '{"note_id":"note_001","participants":["Manager Zhang"]}',
        },
    ]

    normalized = _normalize_dynamic_tool_input(
        "notes_get",
        {"note_id": "<referenced_note_id_from_step2>"},
        tool_results,
        "Get note_004 because it is the previous meeting cross-referenced in note_001.",
    )

    assert normalized["note_id"] == "note_004"


def test_notes_get_previous_step_does_not_mean_previous_meeting() -> None:
    tool_results = [
        {
            "tool": "notes_list",
            "status": "success",
            "result": '{"notes":[{"note_id":"note_001"},{"note_id":"note_002"},{"note_id":"note_004"}]}',
        },
    ]

    normalized = _normalize_dynamic_tool_input(
        "notes_get",
        {"note_id": "<id_from_step1>"},
        tool_results,
        "Get the detailed content of the February 23 weekly meeting note using the note ID identified from the previous step.",
    )

    assert normalized["note_id"] == "note_001"
