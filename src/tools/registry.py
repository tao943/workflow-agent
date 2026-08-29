import json
import os
import urllib.error
import urllib.request
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from src.argument_provenance import ArgumentProvenanceRule
from src.project_files import ProjectFileFilter
from src.tools.calculator import calculator
from src.tools.datetime_tool import datetime_tool
from src.tools.notes_tool import NOTES_FILE, notes_tool


DangerLevel = Literal["safe", "confirm", "dangerous"]
ToolStatus = Literal["success", "error", "denied", "invalid_args", "timeout"]
ProvenancePolicy = Literal["off", "warn", "repair", "require"]


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    step_id: str | None
    output_dir: Path


@dataclass(frozen=True)
class ToolResult:
    title: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attachments: list[str] = field(default_factory=list)
    status: ToolStatus = "success"
    display_output: str | None = None
    raw_output_path: str | None = None
    duration_ms: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    output_schema: dict[str, Any]
    permissions: list[str]
    danger_level: DangerLevel
    examples: list[str]
    timeout_seconds: int
    truncate_policy: dict[str, int]
    execute: Callable[[dict[str, Any], ToolContext], ToolResult]
    provider_name: str = "builtin"
    provider_kind: str = "builtin"
    source: str = "builtin"
    schema_hash: str = ""
    trust_level: str = "project"
    argument_provenance_rules: list[ArgumentProvenanceRule] = field(default_factory=list)
    provenance_policy: ProvenancePolicy = "off"


def _require_text(args: dict[str, Any], key: str = "input") -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"参数 {key} 必须是非空字符串。")
    return value


def _calculator_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    expression = _require_text(args)
    output = calculator(expression)
    return ToolResult(title="计算完成", output=output, metadata={"expression": expression})


def _datetime_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    output = datetime_tool(str(args.get("input", "")))
    return ToolResult(title="当前时间", output=output)


def _notes_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    content = _require_text(args)
    output = notes_tool(content)
    return ToolResult(title="笔记已保存", output=output, attachments=[str(NOTES_FILE)])


def _resolve_workspace_path(context: ToolContext, value: str) -> Path:
    base = Path.cwd().resolve()
    target = (base / value).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("只允许访问当前项目目录内的路径。")
    return target


def _read_file_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    target = _resolve_workspace_path(context, _require_text(args, "path"))
    if not target.exists() or not target.is_file():
        raise ValueError(f"文件不存在：{target}")
    text = target.read_text(encoding="utf-8")
    return ToolResult(title=f"读取文件 {target.name}", output=text, metadata={"path": str(target), "truncated": len(text) > 4000})


def _write_file_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _require_text(args, "path")
    content = _require_text(args, "content")
    target = _resolve_workspace_path(context, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ToolResult(title=f"写入文件 {target.name}", output=f"已写入：{target}", metadata={"path": str(target)}, attachments=[str(target)])


def _apply_patch_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    patch = _require_text(args, "patch")
    if ".." in patch or any(token in patch.lower() for token in (".env", "outputs/", "outputs\\", ".git/", ".git\\")):
        raise ValueError("补丁包含被禁止的运行时、凭据或越界路径。")
    checked = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", "-"], input=patch, text=True, encoding="utf-8", capture_output=True, check=False)
    if checked.returncode:
        raise ValueError(f"补丁校验失败：{checked.stderr[-1000:]}")
    applied = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=patch, text=True, encoding="utf-8", capture_output=True, check=False)
    if applied.returncode:
        raise ValueError(f"补丁应用失败：{applied.stderr[-1000:]}")
    return ToolResult(title="补丁已应用", output="已通过 git apply 校验并应用补丁。", metadata={"patch_bytes": len(patch.encode("utf-8"))})


def _list_files_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    root = _resolve_workspace_path(context, str(args.get("path", ".")))
    if not root.exists() or not root.is_dir():
        raise ValueError(f"目录不存在：{root}")
    files = []
    for item in ProjectFileFilter(Path.cwd()).iter_project_files(root, limit=200):
        files.append(str(item.relative_to(Path.cwd())))
    return ToolResult(title="文件列表", output="\n".join(files), metadata={"count": len(files)})


def _search_files_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    query = _require_text(args, "query")
    root = _resolve_workspace_path(context, str(args.get("path", ".")))
    matches: list[str] = []
    for item in ProjectFileFilter(Path.cwd()).iter_project_files(root):
        try:
            for index, line in enumerate(item.read_text(encoding="utf-8").splitlines(), start=1):
                if query in line:
                    matches.append(f"{item.relative_to(Path.cwd())}:{index}: {line.strip()}")
                if len(matches) >= 100:
                    break
        except UnicodeDecodeError:
            continue
        if len(matches) >= 100:
            break
    return ToolResult(title="搜索结果", output="\n".join(matches) or "未找到匹配内容", metadata={"query": query, "count": len(matches)})


def _rag_search_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    query = _require_text(args, "query")
    top_k = int(args.get("top_k", 5) or 5)
    source_filter = args.get("source_filter")
    service_base = os.getenv("RAG_API_BASE", "").rstrip("/")
    if service_base:
        return _rag_http_search(query, top_k, source_filter if isinstance(source_filter, str) else None, args, service_base)

    knowledge_file = Path("outputs/knowledge_base.json")
    if not knowledge_file.exists():
        return ToolResult(
            title="RAG 未配置",
            output="",
            error="RAG search is not configured. Set RAG_API_BASE or add outputs/knowledge_base.json.",
            status="error",
            metadata={"query": query, "top_k": top_k, "source_filter": source_filter, "code": "not_configured"},
        )

    payload = json.loads(knowledge_file.read_text(encoding="utf-8"))
    documents = payload if isinstance(payload, list) else payload.get("documents", [])
    query_terms = [term for term in query.lower().split() if term]
    results: list[dict[str, Any]] = []
    for index, item in enumerate(documents):
        text = str(item.get("text", ""))
        title = str(item.get("title", f"doc-{index}"))
        path = str(item.get("path", ""))
        if source_filter and source_filter not in title and source_filter not in path:
            continue
        score = sum(1 for term in query_terms if term in text.lower() or term in title.lower())
        if score:
            results.append(
                {
                    "id": str(item.get("id", index)),
                    "title": title,
                    "score": float(score),
                    "path": path,
                    "text": text[:500],
                }
            )
    results = sorted(results, key=lambda row: row["score"], reverse=True)[:top_k]
    sources = [{key: value for key, value in item.items() if key != "text"} for item in results]
    output = "\n\n".join(f"[{item['title']}] {item['text']}" for item in results) or "未找到相关内容。"
    metadata = {"query": query, "top_k": top_k, "sources": sources}
    if not sources:
        metadata["code"] = "no_evidence"
        metadata["diagnostics"] = {"no_evidence": True}
    return ToolResult(title="RAG 检索结果", output=output, metadata=metadata)


def _rag_http_search(query: str, top_k: int, source_filter: str | None, args: dict[str, Any], service_base: str) -> ToolResult:
    user_id = os.getenv("RAG_USER_ID", "user_8be74c46a603")
    knowledge_base_id = os.getenv("RAG_KNOWLEDGE_BASE_ID", "kb_001")
    api_key = os.getenv("RAG_API_KEY") or os.getenv("RAG_SERVICE_API_KEY") or ""
    mode = str(args.get("mode") or os.getenv("RAG_SEARCH_MODE", "answer")).lower()
    endpoint = "/v1/retrieve" if mode == "retrieve" else "/v1/answer"
    payload: dict[str, Any] = {
        "question": query,
        "user_id": user_id,
        "knowledge_base_id": knowledge_base_id,
    }
    if endpoint == "/v1/answer":
        payload["retrieval_options"] = {
            "top_k": top_k,
            "fetch_k": int(args.get("fetch_k", 12) or 12),
            "max_distance": float(args.get("max_distance", 0.7) or 0.7),
            "use_hybrid": bool(args.get("use_hybrid", True)),
            "use_hierarchical": bool(args.get("use_hierarchical", True)),
            "context_enhancement_mode": str(args.get("context_enhancement_mode", "parent")),
            "context_selection_strategy": str(args.get("context_selection_strategy", "balanced")),
            "answer_style": str(args.get("answer_style", "concise")),
        }
    else:
        payload["top_k"] = top_k
    if source_filter:
        payload["source_filter"] = source_filter

    request = urllib.request.Request(
        f"{service_base}{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_rag_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("RAG_TIMEOUT_SECONDS", "120"))) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return ToolResult(
            title="RAG HTTP 调用失败",
            output="",
            error=f"RAG service returned HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:500]}",
            status="error",
            metadata={"query": query, "top_k": top_k, "endpoint": endpoint, "code": "http_error"},
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        return ToolResult(
            title="RAG 服务不可用",
            output="",
            error=f"Cannot connect to RAG service at {service_base}: {exc}",
            status="error",
            metadata={"query": query, "top_k": top_k, "endpoint": endpoint, "code": "service_unavailable"},
        )

    sources = data.get("sources") or []
    diagnostics = data.get("diagnostics") or {}
    if endpoint == "/v1/retrieve":
        output = _format_rag_sources(sources)
    else:
        answer = str(data.get("answer") or "")
        output = answer
        if sources:
            output = answer + "\n\n参考来源：\n" + _format_rag_sources(sources)
        if diagnostics.get("no_evidence"):
            output = "知识库中没有找到足够依据。"
    metadata_code = "no_evidence" if diagnostics.get("no_evidence") or not sources else "ok"
    return ToolResult(
        title="RAG HTTP 检索结果",
        output=output,
        metadata={
            "query": query,
            "top_k": top_k,
            "endpoint": endpoint,
            "user_id": user_id,
            "knowledge_base_id": knowledge_base_id,
            "sources": sources,
            "diagnostics": diagnostics,
            "code": metadata_code,
        },
    )


def _rag_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _format_rag_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "未找到相关来源。"
    lines = []
    for index, item in enumerate(sources, start=1):
        number = item.get("source_number", index)
        source = item.get("source") or item.get("title") or item.get("id") or f"source-{index}"
        location = item.get("location") or item.get("path") or ""
        snippet = item.get("text") or item.get("content") or item.get("snippet") or ""
        if snippet:
            snippet = " ".join(str(snippet).split())[:300]
            lines.append(f"[来源 {number}] {source} {location}\n{snippet}")
        else:
            lines.append(f"[来源 {number}] {source} {location}".strip())
    return "\n".join(lines)


def _team_storage(context: ToolContext):
    from src.storage import Storage

    return Storage(context.output_dir / "agent.sqlite")


def _team_create_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    from src.config import get_team_spec

    name = _require_text(args, "name")
    spec = get_team_spec(name)
    team_id = f"team_{spec.name}"
    _team_storage(context).create_team(team_id, spec.name, spec.description, spec.model_dump())
    return ToolResult(title="Team created", output=f"Created team {spec.name}", metadata={"team_id": team_id, "name": spec.name})


def _team_list_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    teams = _team_storage(context).list_teams()
    output = "\n".join(f"{item['id']} | {item['name']} | {item['status']}" for item in teams) or "No teams."
    return ToolResult(title="Teams", output=output, metadata={"teams": teams})


def _team_status_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    storage = _team_storage(context)
    run = storage.get_team_run(team_run_id)
    if not run:
        return ToolResult(title="Team status", output="", error=f"Unknown team run: {team_run_id}", status="error")
    return ToolResult(title="Team status", output=f"{run['id']} | {run['team_id']} | {run['status']}", metadata={"run": run, "members": storage.list_team_members(team_run_id), "tasks": storage.list_team_tasks(team_run_id)})


def _team_delete_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_id = _require_text(args, "team_id")
    deleted = _team_storage(context).delete_team(team_id)
    if not deleted:
        return ToolResult(title="Team delete", output="", error=f"Cannot delete active or unknown team: {team_id}", status="error")
    return ToolResult(title="Team deleted", output=f"Deleted {team_id}", metadata={"team_id": team_id})


def _team_send_message_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    sender = _require_text(args, "sender")
    recipient = _require_text(args, "recipient")
    content = _require_text(args, "content")
    if len(content) > 12000:
        return ToolResult(title="Team message", output="", error="Message payload is too long.", status="invalid_args")
    from src.session import new_id

    _team_storage(context).add_team_message(new_id("team_msg"), team_run_id, sender, recipient, {"type": "message", "content": content})
    return ToolResult(title="Team message sent", output=f"{sender} -> {recipient}: {content[:200]}", metadata={"team_run_id": team_run_id})


def _team_task_create_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    title = _require_text(args, "title")
    description = _require_text(args, "description")
    assigned_to = args.get("assigned_to")
    from src.session import new_id

    task_id = new_id("team_task")
    _team_storage(context).add_team_task(task_id, team_run_id, title, description, assigned_to if isinstance(assigned_to, str) else None)
    return ToolResult(title="Team task created", output=f"Created task {task_id}: {title}", metadata={"task_id": task_id})


def _team_task_list_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    tasks = _team_storage(context).list_team_tasks(team_run_id)
    output = "\n".join(f"{item['id']} | {item['status']} | {item['assigned_to']} | {item['title']}" for item in tasks) or "No tasks."
    return ToolResult(title="Team tasks", output=output, metadata={"tasks": tasks})


def _team_task_get_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    task_id = _require_text(args, "task_id")
    task = _team_storage(context).get_team_task(task_id)
    if not task:
        return ToolResult(title="Team task", output="", error=f"Unknown task: {task_id}", status="error")
    return ToolResult(title="Team task", output=f"{task['id']} | {task['status']} | {task['title']}\n{task['description']}\n{task.get('result') or ''}", metadata={"task": task})


def _team_task_update_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    task_id = _require_text(args, "task_id")
    status = _require_text(args, "status")
    result = args.get("result")
    assigned_to = args.get("assigned_to")
    _team_storage(context).update_team_task(task_id, status, result if isinstance(result, str) else None, assigned_to if isinstance(assigned_to, str) else None)
    return ToolResult(title="Team task updated", output=f"Updated {task_id} -> {status}", metadata={"task_id": task_id, "status": status})


def _team_shutdown_request_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    reason = str(args.get("reason", "shutdown requested"))
    return ToolResult(title="Team shutdown requested", output=reason, metadata={"team_run_id": team_run_id, "requested": True})


def _team_approve_shutdown_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    _team_storage(context).update_team_run(team_run_id, "completed")
    return ToolResult(title="Team shutdown approved", output=f"Approved shutdown for {team_run_id}", metadata={"team_run_id": team_run_id})


def _team_reject_shutdown_execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
    team_run_id = _require_text(args, "team_run_id")
    return ToolResult(title="Team shutdown rejected", output=f"Rejected shutdown for {team_run_id}", metadata={"team_run_id": team_run_id})


OUTPUT_SCHEMA = {
    "title": "str",
    "output": "str",
    "metadata": "dict",
    "error": "str | None",
    "attachments": "list[str]",
}


TOOL_REGISTRY = {
    "calculator": ToolSpec(
        name="calculator",
        description="执行安全的基础数学表达式计算",
        parameters={"input": "数学表达式字符串，例如：12 * (8 + 5)"},
        output_schema=OUTPUT_SCHEMA,
        permissions=[],
        danger_level="safe",
        examples=["12 * (8 + 5)"],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_calculator_execute,
    ),
    "datetime_tool": ToolSpec(
        name="datetime_tool",
        description="获取当前本地日期和时间",
        parameters={"input": "任意说明文本"},
        output_schema=OUTPUT_SCHEMA,
        permissions=[],
        danger_level="safe",
        examples=["今天日期"],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_datetime_execute,
    ),
    "notes_tool": ToolSpec(
        name="notes_tool",
        description="把内容保存到项目 outputs/notes.md 文件",
        parameters={"input": "要保存成笔记的 Markdown 文本"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["write:outputs/notes.md"],
        danger_level="confirm",
        examples=["保存当前计划"],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_notes_execute,
    ),
    "read_file": ToolSpec(
        name="read_file",
        description="读取项目内文本文件",
        parameters={"path": "项目内相对路径"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["read:*"],
        danger_level="safe",
        examples=["README.md"],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_read_file_execute,
    ),
    "write_file": ToolSpec(
        name="write_file",
        description="写入项目内文本文件",
        parameters={"path": "项目内相对路径", "content": "文件内容"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["write:*"],
        danger_level="confirm",
        examples=["outputs/example.md"],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_write_file_execute,
        argument_provenance_rules=[
            ArgumentProvenanceRule(
                target_tool="write_file",
                target_arg="path",
                source_tool="safe_default",
                source_field="outputs/",
                match_mode="path_under",
                failure_action="warn",
                source_type="safe_default",
                source_value="outputs/",
            )
        ],
        provenance_policy="warn",
    ),
    "apply_patch": ToolSpec(
        name="apply_patch",
        description="在当前项目工作区安全校验并应用 unified diff 补丁",
        parameters={"patch": "unified diff 文本"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["write:patch"],
        danger_level="confirm",
        examples=["*** Begin Patch\n*** Update File: src/example.py"],
        timeout_seconds=30,
        truncate_policy={"max_chars": 4000},
        execute=_apply_patch_execute,
    ),
    "list_files": ToolSpec(
        name="list_files",
        description="列出项目内文件",
        parameters={"path": "项目内相对路径，默认当前目录"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["read:*"],
        danger_level="safe",
        examples=["."],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_list_files_execute,
    ),
    "search_files": ToolSpec(
        name="search_files",
        description="搜索项目内文本文件",
        parameters={"query": "搜索关键词", "path": "项目内相对路径，默认当前目录"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["read:*"],
        danger_level="safe",
        examples=["LangGraph"],
        timeout_seconds=5,
        truncate_policy={"max_chars": 4000},
        execute=_search_files_execute,
    ),
    "rag_search": ToolSpec(
        name="rag_search",
        description="调用本地 RAG HTTP 服务或 fallback 知识库检索相关内容",
        parameters={"query": "检索查询"},
        output_schema=OUTPUT_SCHEMA,
        permissions=["rag:*"],
        danger_level="confirm",
        examples=["LangGraph checkpoint 怎么用"],
        timeout_seconds=15,
        truncate_policy={"max_chars": 4000},
        execute=_rag_search_execute,
    ),
    "team_create": ToolSpec("team_create", "Create a configured agent team.", {"name": "Team name"}, OUTPUT_SCHEMA, ["team:create:*"], "confirm", ["default-dev-team"], 5, {"max_chars": 4000}, _team_create_execute),
    "team_list": ToolSpec("team_list", "List configured teams.", {}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_list_execute),
    "team_status": ToolSpec("team_status", "Show team run status.", {"team_run_id": "Team run id"}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_status_execute),
    "team_delete": ToolSpec("team_delete", "Delete an inactive team.", {"team_id": "Team id"}, OUTPUT_SCHEMA, ["team:delete:*"], "confirm", [], 5, {"max_chars": 4000}, _team_delete_execute),
    "team_send_message": ToolSpec("team_send_message", "Send a team mailbox message.", {"team_run_id": "Team run id", "sender": "Sender", "recipient": "Recipient", "content": "Message content"}, OUTPUT_SCHEMA, ["team:message:*"], "safe", [], 5, {"max_chars": 4000}, _team_send_message_execute),
    "team_task_create": ToolSpec("team_task_create", "Create a team task.", {"team_run_id": "Team run id", "title": "Title", "description": "Description", "assigned_to": "Optional assignee"}, OUTPUT_SCHEMA, ["team:task:*"], "safe", [], 5, {"max_chars": 4000}, _team_task_create_execute),
    "team_task_list": ToolSpec("team_task_list", "List team tasks.", {"team_run_id": "Team run id"}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_task_list_execute),
    "team_task_get": ToolSpec("team_task_get", "Get a team task.", {"task_id": "Task id"}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_task_get_execute),
    "team_task_update": ToolSpec("team_task_update", "Update a team task.", {"task_id": "Task id", "status": "Status", "result": "Optional result", "assigned_to": "Optional assignee"}, OUTPUT_SCHEMA, ["team:task:*"], "safe", [], 5, {"max_chars": 4000}, _team_task_update_execute),
    "team_shutdown_request": ToolSpec("team_shutdown_request", "Request team shutdown.", {"team_run_id": "Team run id", "reason": "Reason"}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_shutdown_request_execute),
    "team_approve_shutdown": ToolSpec("team_approve_shutdown", "Approve team shutdown.", {"team_run_id": "Team run id"}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_approve_shutdown_execute),
    "team_reject_shutdown": ToolSpec("team_reject_shutdown", "Reject team shutdown.", {"team_run_id": "Team run id"}, OUTPUT_SCHEMA, [], "safe", [], 5, {"max_chars": 4000}, _team_reject_shutdown_execute),
}


def get_tool_names() -> set[str]:
    return set(TOOL_REGISTRY)


def register_tool(tool: ToolSpec, replace: bool = False) -> None:
    if tool.name in TOOL_REGISTRY and not replace:
        raise ValueError(f"Tool already registered: {tool.name}")
    TOOL_REGISTRY[tool.name] = tool


def register_tools(tools: list[ToolSpec], replace: bool = False) -> None:
    for tool in tools:
        register_tool(tool, replace=replace)


def execute_tool(name: str, args: dict[str, Any], context: ToolContext) -> ToolResult:
    tool = TOOL_REGISTRY[name]
    try:
        result = tool.execute(args, context)
        if result.display_output is None:
            return ToolResult(
                title=result.title,
                output=result.output,
                metadata=result.metadata,
                error=result.error,
                attachments=result.attachments,
                status=result.status if result.status != "success" else ("error" if result.error else result.status),
                display_output=result.output,
                raw_output_path=result.raw_output_path,
                duration_ms=result.duration_ms,
                truncated=result.truncated,
            )
        return result
    except Exception as exc:
        return ToolResult(
            title=f"{name} 调用失败",
            output="",
            error=f"The {name} tool was called with invalid arguments or failed: {exc}. Please rewrite the input.",
            status="invalid_args",
            metadata={"tool": name},
        )
