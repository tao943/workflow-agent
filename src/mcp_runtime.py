from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.config import AppConfig, MCPServerConfig
from src.tools.registry import OUTPUT_SCHEMA, ToolContext, ToolResult, ToolSpec, register_tools


@dataclass(frozen=True)
class MCPToolInfo:
    server: str
    name: str
    local_name: str
    description: str
    input_schema: dict[str, Any]
    schema_hash: str
    trust_level: str


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    enabled: bool
    ok: bool
    transport: str
    tools: list[str] = field(default_factory=list)
    error: str = ""


class MCPClientManager:
    def __init__(
        self,
        config: AppConfig,
        client_factory: Callable[[str, MCPServerConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self.client_factory = client_factory
        self._tool_cache: dict[str, list[MCPToolInfo]] = {}
        self._errors: dict[str, str] = {}

    def discover_tools(self, refresh: bool = False) -> list[MCPToolInfo]:
        if not self.config.mcp.enabled:
            return []
        tools: list[MCPToolInfo] = []
        for name, server in self.config.mcp.servers.items():
            if not server.enabled:
                continue
            if not refresh and name in self._tool_cache:
                tools.extend(self._tool_cache[name])
                continue
            try:
                discovered = self._discover_server_tools(name, server)
                self._tool_cache[name] = discovered
                self._errors.pop(name, None)
                tools.extend(discovered)
            except Exception as exc:
                self._errors[name] = str(exc)
                self._tool_cache[name] = []
        return tools

    def health(self) -> list[MCPServerStatus]:
        statuses: list[MCPServerStatus] = []
        for name, server in self.config.mcp.servers.items():
            if not server.enabled:
                statuses.append(MCPServerStatus(name, False, False, server.transport, [], "disabled"))
                continue
            tools = self._tool_cache.get(name)
            if tools is None:
                self.discover_tools(refresh=True)
                tools = self._tool_cache.get(name, [])
            error = self._errors.get(name, "")
            statuses.append(MCPServerStatus(name, True, not error, server.transport, [item.name for item in tools], error))
        return statuses

    def call_tool(self, server_name: str, tool_name: str, args: dict[str, Any], context: ToolContext) -> ToolResult:
        if server_name not in self.config.mcp.servers:
            return ToolResult(
                title=f"MCP {server_name} failed",
                output="",
                error=f"Unknown MCP server: {server_name}",
                status="invalid_args",
                metadata={"code": "mcp_server_not_found", "mcp_server": server_name, "mcp_tool": tool_name},
            )
        server = self.config.mcp.servers[server_name]
        if not self._tool_allowed(server, tool_name):
            return ToolResult(
                title=f"MCP {tool_name} denied",
                output="",
                error=f"MCP tool is not allowed by server config: {server_name}/{tool_name}",
                status="denied",
                metadata={"code": "mcp_tool_not_allowed", "mcp_server": server_name, "mcp_tool": tool_name},
            )
        try:
            result = self._call_server_tool(server_name, server, tool_name, args)
        except Exception as exc:
            return ToolResult(
                title=f"MCP {tool_name} failed",
                output="",
                error=f"MCP tool call failed: {exc}",
                status="error",
                metadata={"code": "mcp_tool_failed", "mcp_server": server_name, "mcp_tool": tool_name},
            )
        return result

    def register_tools(self, replace: bool = True) -> list[MCPToolInfo]:
        tools = self.configured_tool_infos()
        specs = [self._tool_spec(info) for info in tools]
        register_tools(specs, replace=replace)
        return tools

    def configured_tool_infos(self) -> list[MCPToolInfo]:
        infos: list[MCPToolInfo] = []
        for server_name, server in self.config.mcp.servers.items():
            if not self.config.mcp.enabled or not server.enabled:
                continue
            for tool_name in server.allow_tools:
                if tool_name == "*" or any(fnmatch.fnmatch(tool_name, pattern) for pattern in server.deny_tools):
                    continue
                schema = _default_schema(server_name, tool_name)
                infos.append(
                    MCPToolInfo(
                        server=server_name,
                        name=tool_name,
                        local_name=f"mcp.{server_name}.{tool_name}",
                        description=f"Configured MCP tool {server_name}/{tool_name}",
                        input_schema=schema,
                        schema_hash=_hash_json(schema),
                        trust_level=server.trust_level,
                    )
                )
        return infos

    def _discover_server_tools(self, name: str, server: MCPServerConfig) -> list[MCPToolInfo]:
        raw_tools = self._run_async(asyncio.wait_for(self._list_tools_async(name, server), timeout=self.config.mcp.connect_timeout_seconds))
        result: list[MCPToolInfo] = []
        for raw_tool in raw_tools:
            tool_name = str(getattr(raw_tool, "name", "") or "")
            if not tool_name or not self._tool_allowed(server, tool_name):
                continue
            schema = _model_to_dict(getattr(raw_tool, "inputSchema", None) or getattr(raw_tool, "input_schema", None) or {})
            schema_hash = _hash_json(schema)
            result.append(
                MCPToolInfo(
                    server=name,
                    name=tool_name,
                    local_name=f"mcp.{name}.{tool_name}",
                    description=str(getattr(raw_tool, "description", "") or f"MCP tool {name}/{tool_name}"),
                    input_schema=schema,
                    schema_hash=schema_hash,
                    trust_level=server.trust_level,
                )
            )
        return result

    async def _list_tools_async(self, name: str, server: MCPServerConfig):
        client = self._client(name, server)
        async with client as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return list(getattr(tools_result, "tools", []) or [])

    def _call_server_tool(self, name: str, server: MCPServerConfig, tool_name: str, args: dict[str, Any]) -> ToolResult:
        return self._run_async(asyncio.wait_for(self._call_tool_async(name, server, tool_name, args), timeout=self.config.mcp.tool_timeout_seconds))

    async def _call_tool_async(self, name: str, server: MCPServerConfig, tool_name: str, args: dict[str, Any]) -> ToolResult:
        client = self._client(name, server)
        async with client as session:
            await session.initialize()
            raw_result = await session.call_tool(tool_name, args)
            output, attachments = _mcp_result_to_text(raw_result)
            return ToolResult(
                title=f"MCP {name}/{tool_name}",
                output=output,
                attachments=attachments,
                metadata={
                    "mcp_server": name,
                    "mcp_tool": tool_name,
                    "schema_hash": self._schema_hash(name, tool_name),
                    "trust_level": server.trust_level,
                },
                status="error" if getattr(raw_result, "isError", False) else "success",
            )

    def _client(self, name: str, server: MCPServerConfig):
        if self.client_factory:
            return self.client_factory(name, server)
        return _SDKClientContext(server, self.config.mcp.connect_timeout_seconds)

    def _tool_spec(self, info: MCPToolInfo) -> ToolSpec:
        def execute(args: dict[str, Any], context: ToolContext) -> ToolResult:
            if info.server == "fetch":
                url = str(args.get("url") or "")
                if url and not (url.startswith("http://") or url.startswith("https://")):
                    return ToolResult(
                        title="MCP fetch invalid URL",
                        output="",
                        error="invalid URL: fetch MCP only allows http:// or https:// URLs.",
                        status="invalid_args",
                        metadata={"mcp_server": info.server, "mcp_tool": info.name, "code": "invalid_url"},
                    )
            return self.call_tool(info.server, info.name, args, context)

        return ToolSpec(
            name=info.local_name,
            description=f"[MCP:{info.server}] {info.description}",
            parameters=info.input_schema or {"type": "object", "properties": {}},
            output_schema=OUTPUT_SCHEMA,
            permissions=[f"mcp_server:{info.server}", f"mcp_tool:{info.server}/{info.name}"],
            danger_level="confirm",
            examples=[],
            timeout_seconds=self.config.mcp.tool_timeout_seconds,
            truncate_policy={"max_chars": 4000},
            execute=execute,
        )

    def _schema_hash(self, server: str, tool_name: str) -> str:
        for item in self._tool_cache.get(server, []):
            if item.name == tool_name:
                return item.schema_hash
        return ""

    def _tool_allowed(self, server: MCPServerConfig, tool_name: str) -> bool:
        if any(fnmatch.fnmatch(tool_name, pattern) for pattern in server.deny_tools):
            return False
        return any(fnmatch.fnmatch(tool_name, pattern) for pattern in server.allow_tools)

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("MCP runtime cannot run inside an existing event loop in sync mode.")


class _SDKClientContext:
    def __init__(self, server: MCPServerConfig, timeout_seconds: int) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds
        self._stdio_cm = None
        self._session_cm = None
        self._session = None

    async def __aenter__(self):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("The mcp package is not installed. Run: pip install -r requirements.txt") from exc
        if self.server.transport != "stdio":
            raise RuntimeError(f"Unsupported MCP transport in v1: {self.server.transport}")
        params = StdioServerParameters(
            command=self.server.command,
            args=self.server.args,
            env=_expand_env(self.server.env),
            cwd=_safe_cwd(self.server.cwd),
        )
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(exc_type, exc, tb)
            except RuntimeError as close_error:
                if "different task" not in str(close_error):
                    raise


def load_and_register_mcp_tools(config: AppConfig, storage: Any | None = None) -> MCPClientManager:
    manager = MCPClientManager(config)
    manager.register_tools(replace=True)
    if storage:
        for status in manager.health():
            storage.add_event(
                "mcp",
                "mcp.server.connected" if status.ok else "mcp.server.failed",
                status.__dict__,
            )
    return manager


def _mcp_result_to_text(raw_result: Any) -> tuple[str, list[str]]:
    content = list(getattr(raw_result, "content", []) or [])
    lines: list[str] = []
    attachments: list[str] = []
    for item in content:
        item_type = getattr(item, "type", "")
        if item_type == "text" or hasattr(item, "text"):
            lines.append(str(getattr(item, "text", "")))
        elif item_type in {"image", "audio"}:
            data = getattr(item, "data", "")
            mime = getattr(item, "mimeType", "") or getattr(item, "mime_type", "")
            lines.append(f"[{item_type} content: {mime}, {len(str(data))} chars]")
        elif hasattr(item, "resource"):
            resource = getattr(item, "resource")
            uri = str(getattr(resource, "uri", ""))
            if uri:
                attachments.append(uri)
            lines.append(f"[resource: {uri}]")
        else:
            lines.append(json.dumps(_model_to_dict(item), ensure_ascii=False))
    return "\n".join(line for line in lines if line).strip(), attachments


def _model_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _hash_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _default_schema(server_name: str, tool_name: str) -> dict[str, Any]:
    if server_name == "fetch" or tool_name == "fetch":
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_length": {"type": "integer"},
                "start_index": {"type": "integer"},
            },
            "required": ["url"],
        }
    if "search" in tool_name:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["query"],
        }
    return {"type": "object", "properties": {}}


def _expand_env(env: dict[str, str]) -> dict[str, str]:
    expanded: dict[str, str] = dict(os.environ)
    default_uv_cache = Path("outputs") / "uv-cache"
    default_uv_tool = Path("outputs") / "uv-tools"
    default_uv_cache.mkdir(parents=True, exist_ok=True)
    default_uv_tool.mkdir(parents=True, exist_ok=True)
    expanded.setdefault("UV_CACHE_DIR", str(default_uv_cache.resolve()))
    expanded.setdefault("UV_TOOL_DIR", str(default_uv_tool.resolve()))
    for key, value in env.items():
        if value.startswith("${") and value.endswith("}"):
            expanded[key] = os.getenv(value[2:-1], "")
        else:
            expanded[key] = value
    return expanded


def _safe_cwd(raw_cwd: str) -> str:
    root = Path.cwd().resolve()
    target = (root / raw_cwd).resolve()
    if root != target and root not in target.parents:
        raise RuntimeError("MCP server cwd must stay inside the project directory.")
    return str(target)
