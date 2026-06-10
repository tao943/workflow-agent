from src.tool_intent import required_tool_requirements, required_tools_for_task, user_intends_tool


def test_required_tools_detect_file_operations():
    assert required_tools_for_task("列出当前项目的主要文件结构", "research") == ["list_files"]
    assert required_tools_for_task("搜索项目中 AgentRuntime 的定义位置", "research") == ["search_files"]
    assert required_tools_for_task("读取 README 并总结这个项目", "research") == ["read_file"]


def test_required_tools_detect_project_analysis():
    assert required_tools_for_task("分析当前项目结构并给出优化建议", "research") == ["list_files", "read_file", "search_files"]
    requirements = required_tool_requirements("分析当前项目结构并给出优化建议", "research")
    assert {"tool": "list_files", "reason": "user asked for project or file structure"} in requirements


def test_required_tools_detect_write_and_rag():
    assert required_tools_for_task("把 hello benchmark 写入 outputs/benchmark_note.txt", "build") == ["write_file"]
    assert required_tools_for_task("检索知识库中的 LangGraph checkpoint", "research") == ["rag_search"]


def test_external_directory_write_is_not_routed_to_write_file():
    assert required_tools_for_task("把结果写入 C:/Users/Public/out.txt", "build") == []


def test_plan_agent_does_not_require_write_tools():
    assert required_tools_for_task("帮我保存一段计划到笔记", "plan") == []


def test_hyphenated_tool_calling_does_not_trigger_calculator():
    assert "calculator" not in required_tools_for_task("解释 tool-calling 和 function-calling 的区别", "plan")
    assert not user_intends_tool("解释 tool-calling 和 function-calling 的区别", "calculator")


def test_math_expression_requires_math_intent():
    assert required_tools_for_task("计算 12 + 30", "build") == ["calculator"]
    assert required_tools_for_task("版本 12-30 的说明", "plan") == []
