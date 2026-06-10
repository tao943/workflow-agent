from src.nodes.planner import _normalize_steps


def test_planner_compacts_many_content_steps():
    steps = _normalize_steps(
        [
            {"description": "第一天计划", "tool_name": None},
            {"description": "第二天计划", "tool_name": None},
            {"description": "第三天计划", "tool_name": None},
            {"description": "整理笔记文本", "tool_name": None},
            {"description": "总结", "tool_name": None},
        ],
        enabled_tools=set(),
    )

    assert [step["description"] for step in steps] == ["生成完整结果"]


def test_planner_compacts_three_content_steps():
    steps = _normalize_steps(
        [
            {"description": "制定三天学习计划", "tool_name": None},
            {"description": "整理笔记文本", "tool_name": None},
            {"description": "说明保存限制", "tool_name": None},
        ],
        enabled_tools=set(),
    )

    assert [step["description"] for step in steps] == ["生成完整结果"]


def test_planner_keeps_tool_steps_when_compacting():
    steps = _normalize_steps(
        [
            {"description": "第一天计划", "tool_name": None},
            {"description": "第二天计划", "tool_name": None},
            {"description": "第三天计划", "tool_name": None},
            {"description": "保存笔记", "tool_name": "notes_tool"},
        ],
        enabled_tools={"notes_tool"},
    )

    assert [step["description"] for step in steps] == ["生成完整结果", "保存笔记"]


def test_planner_filters_unrequested_datetime_tool():
    steps = _normalize_steps(
        [
            {"description": "生成完整结果", "tool_name": None},
            {"description": "查询当前日期时间，确定计划起始日", "tool_name": "datetime_tool"},
        ],
        enabled_tools={"datetime_tool"},
        user_task="帮我制定一个三天学习 LangGraph 的计划",
    )

    assert [step["description"] for step in steps] == ["生成完整结果"]
    assert all(step["tool_name"] is None for step in steps)


def test_planner_keeps_requested_datetime_tool():
    steps = _normalize_steps(
        [
            {"description": "生成完整结果", "tool_name": None},
            {"description": "查询当前日期时间，确定计划起始日", "tool_name": "datetime_tool"},
        ],
        enabled_tools={"datetime_tool"},
        user_task="从今天开始，帮我制定一个三天学习 LangGraph 的计划",
    )

    assert [step["tool_name"] for step in steps] == [None, "datetime_tool"]


def test_planner_preserves_structured_step_metadata():
    steps = _normalize_steps(
        [
            {
                "description": "保存笔记",
                "tool_name": "notes_tool",
                "intent": "write_note",
                "expected_output": "outputs/notes.md 中存在笔记",
                "done_criteria": "notes_tool 返回成功",
                "tool_reason": "用户要求保存",
                "risk_level": "medium",
            }
        ],
        enabled_tools={"notes_tool"},
        user_task="帮我保存为笔记",
    )

    assert steps[0]["intent"] == "write_note"
    assert steps[0]["requires_tool"] is True
    assert steps[0]["expected_output"] == "outputs/notes.md 中存在笔记"
    assert steps[0]["done_criteria"] == "notes_tool 返回成功"
    assert steps[0]["tool_reason"] == "用户要求保存"
    assert steps[0]["risk_level"] == "medium"
