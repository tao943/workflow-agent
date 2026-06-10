import re


READ_KEYWORDS = (
    "读取",
    "读",
    "查看文件",
    "打开文件",
    "总结文件",
    "summarize file",
    "read",
    "readme",
)
LIST_KEYWORDS = (
    "列出",
    "目录",
    "文件结构",
    "项目结构",
    "文件列表",
    "有哪些文件",
    "project structure",
    "list files",
)
SEARCH_KEYWORDS = (
    "搜索",
    "查找",
    "定义位置",
    "在哪里定义",
    "定位",
    "search",
    "find",
)
WRITE_KEYWORDS = (
    "写入",
    "写到",
    "输出到",
    "保存到",
    "保存为",
    "创建文件",
    "write",
    "save",
)
NOTES_KEYWORDS = ("笔记", "记录", "note", "notes")
RAG_KEYWORDS = ("知识库", "rag", "检索")
DATETIME_KEYWORDS = ("时间", "日期", "今天", "现在", "明天", "昨天", "几点", "星期")
MATH_KEYWORDS = ("计算", "算", "数学", "calculate")
PROJECT_ANALYSIS_KEYWORDS = ("分析项目", "项目分析", "项目结构", "分析代码", "结构分析", "优化建议")


READ_KEYWORDS = ("读取", "查看文件", "打开文件", "总结文件", "read", "readme", "summarize file")
LIST_KEYWORDS = ("列出", "目录", "文件结构", "项目结构", "文件列表", "有哪些文件", "project structure", "list files")
SEARCH_KEYWORDS = ("搜索", "查找", "定义位置", "在哪里定义", "定位", "search", "find")
WRITE_KEYWORDS = ("写入", "写到", "输出到", "保存到", "保存为", "创建文件", "write", "save")
NOTES_KEYWORDS = ("笔记", "记录", "note", "notes")
RAG_KEYWORDS = ("知识库", "rag", "检索")
DATETIME_KEYWORDS = ("时间", "日期", "今天", "现在", "明天", "昨天", "几点", "星期")
MATH_KEYWORDS = ("计算", "算", "数学", "calculate")
PROJECT_ANALYSIS_KEYWORDS = (
    "分析项目",
    "项目分析",
    "项目结构",
    "分析代码",
    "结构分析",
    "优化建议",
    "测试体系",
    "benchmark",
    "deepeval",
    "claw",
)


def required_tools_for_task(user_task: str, agent: str | None = None) -> list[str]:
    """Return deterministic tool requirements implied by the user task.

    This is intentionally conservative: it detects direct intent and leaves
    broader strategy to the planner/router. The same function is used by
    planner fallback and verifier so tool expectations stay aligned.
    """

    text = _norm(user_task)
    tools: list[str] = []
    wants_write = _contains_any(text, WRITE_KEYWORDS)

    if _has_math_expression(user_task) and _contains_any(text, MATH_KEYWORDS):
        tools.append("calculator")
    if _contains_any(text, DATETIME_KEYWORDS):
        tools.append("datetime_tool")

    project_analysis = _contains_any(text, PROJECT_ANALYSIS_KEYWORDS) and not wants_write
    if project_analysis or _contains_any(text, LIST_KEYWORDS):
        tools.append("list_files")
    if project_analysis or _contains_any(text, READ_KEYWORDS):
        tools.append("read_file")
    if project_analysis or _contains_any(text, SEARCH_KEYWORDS):
        tools.append("search_files")

    if _contains_any(text, RAG_KEYWORDS):
        tools.append("rag_search")

    if wants_write and agent != "plan":
        if _mentions_external_path(user_task):
            return _dedupe(tools)
        if _mentions_outputs_path(user_task):
            tools.append("write_file")
        elif _contains_any(text, NOTES_KEYWORDS):
            tools.append("notes_tool")
        else:
            tools.append("write_file")
    elif _contains_any(text, NOTES_KEYWORDS) and _contains_any(text, WRITE_KEYWORDS) and agent != "plan":
        tools.append("notes_tool")

    return _dedupe(tools)


def required_tool_requirements(user_task: str, agent: str | None = None) -> list[dict[str, str]]:
    reasons = {
        "calculator": "user asked for a calculation",
        "datetime_tool": "user asked for date or time",
        "list_files": "user asked for project or file structure",
        "read_file": "user asked to read or summarize files",
        "search_files": "user asked to search, locate, or analyze code",
        "rag_search": "user asked to search the knowledge base",
        "write_file": "user asked to write or save an output file",
        "notes_tool": "user asked to save notes",
    }
    return [{"tool": tool, "reason": reasons.get(tool, "required by task intent")} for tool in required_tools_for_task(user_task, agent)]


def user_intends_tool(user_task: str, tool_name: str | None) -> bool:
    if tool_name is None:
        return True
    if not user_task.strip():
        return True
    if tool_name.startswith("team_"):
        text = _norm(user_task)
        return _contains_any(text, ("team", "团队", "多 agent", "multi-agent"))
    return tool_name in required_tools_for_task(user_task)


def _norm(text: str) -> str:
    return text.lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _has_math_expression(text: str) -> bool:
    return bool(re.search(r"\d+\s*[-+*/]\s*\d+", text))


def _mentions_outputs_path(text: str) -> bool:
    return bool(re.search(r"outputs[\\/][\w./\\-]+\.(?:md|txt|py|json|csv)", text, re.I))


def _mentions_external_path(text: str) -> bool:
    return bool(re.search(r"\b[A-Za-z]:[\\/](?!.*outputs[\\/])", text))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
