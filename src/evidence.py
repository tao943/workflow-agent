from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


TOOL_BYPASS_TERMS = {
    "list_files": ["已扫描", "扫描项目", "文件树", "目录结构", "file tree", "listed files"],
    "read_file": ["已读取", "读取 README", "读取文件", "read file"],
    "search_files": ["已搜索", "搜索项目", "检索代码", "searched files"],
    "notes_tool": ["已保存", "保存为笔记", "写入笔记"],
    "write_file": ["已写入", "写入文件", "created file"],
    "rag_search": ["已检索", "检索知识库", "RAG 检索"],
}


@dataclass
class EvidenceCheck:
    passed: bool
    status: str
    issues: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def verify_task_evidence(task: dict[str, Any], member_result: dict[str, Any], tool_calls: list[dict[str, Any]]) -> EvidenceCheck:
    issues: list[str] = []
    required_tools = list(task.get("required_tools") or [])
    called_tools = {item.get("tool_name") for item in tool_calls if not item.get("error")}

    for tool in required_tools:
        if tool not in called_tools:
            issues.append(f"required tool was not called: {tool}")

    answer = str(member_result.get("final_answer") or "")
    for tool in required_tools:
        terms = TOOL_BYPASS_TERMS.get(tool, [])
        if tool not in called_tools and any(term.lower() in answer.lower() for term in terms):
            issues.append(f"tool bypass suspected: answer claims {tool}-like work without tool evidence")

    if not required_tools:
        status = "passed" if not issues else "missing"
    elif not issues:
        status = "passed"
    elif called_tools.intersection(required_tools):
        status = "partial"
    else:
        status = "missing"

    return EvidenceCheck(passed=not issues, status=status, issues=issues, tool_calls=tool_calls)


def verify_final_answer_evidence(final_answer: str, evidence_ids: set[str]) -> EvidenceCheck:
    issues: list[str] = []
    section_groups = [
        ["## 当前项目结构分析", "## 褰撳墠椤圭洰缁撴瀯鍒嗘瀽"],
        ["## 优化建议", "## 浼樺寲寤鸿"],
    ]
    for group in section_groups:
        if not any(section in final_answer for section in group):
            issues.append(f"missing final answer section: {group[0]}")

    if "Member results" in final_answer and not any(section in final_answer for section in ("## 优化建议", "## 浼樺寲寤鸿")):
        issues.append("final answer only contains member status, not user-facing analysis")

    referenced_ids = set(re.findall(r"\bev_\d+\b", final_answer))
    for token in final_answer.replace("，", " ").replace(",", " ").split():
        if token.startswith("ev_"):
            referenced_ids.add(token.strip("。;；，,"))

    if not referenced_ids:
        issues.append("final answer does not reference any evidence id")

    missing = sorted(referenced_ids - evidence_ids)
    for evidence_id in missing:
        issues.append(f"final answer references missing evidence id: {evidence_id}")

    return EvidenceCheck(passed=not issues, status="passed" if not issues else "missing", issues=issues)
