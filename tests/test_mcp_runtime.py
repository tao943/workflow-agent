from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.config import AppConfig, MCPConfig, MCPServerConfig, PermissionRule
from src.mcp_runtime import MCPClientManager
from src.storage import Storage
from src.tool_runtime import ToolRuntime, ToolRunRequest
from src.tools.registry import TOOL_REGISTRY, ToolContext


class FakeMCPClient:
    def __init__(self, tools=None, output="ok"):
        self.tools = tools or [
            SimpleNamespace(
                name="fetch",
                description="Fetch a URL",
                inputSchema={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            )
        ]
        self.output = output

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, args):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"{self.output}:{name}:{args}")], isError=False)


def _config(enabled=True):
    return AppConfig(
        output_dir=str(Path("outputs") / "test_mcp" / uuid4().hex[:8]),
        mcp=MCPConfig(
            enabled=enabled,
            servers={
                "fetch": MCPServerConfig(
                    enabled=True,
                    command="fake",
                    args=[],
                    allow_tools=["fetch"],
                    trust_level="project",
                )
            },
        ),
    )


def test_mcp_disabled_does_not_discover_tools():
    manager = MCPClientManager(_config(enabled=False), client_factory=lambda name, server: FakeMCPClient())

    assert manager.discover_tools() == []


def test_fake_stdio_mcp_server_discovers_and_namespaces_tool():
    manager = MCPClientManager(_config(), client_factory=lambda name, server: FakeMCPClient())

    tools = manager.discover_tools(refresh=True)

    assert tools[0].local_name == "mcp.fetch.fetch"
    assert tools[0].schema_hash


def test_mcp_registers_tool_spec_with_schema():
    config = _config()
    manager = MCPClientManager(config, client_factory=lambda name, server: FakeMCPClient())
    manager.register_tools()

    spec = TOOL_REGISTRY["mcp.fetch.fetch"]

    assert spec.parameters["required"] == ["url"]
    assert "mcp_tool:fetch/fetch" in spec.permissions


def test_mcp_tool_invalid_args_are_structured():
    config = _config()
    manager = MCPClientManager(config, client_factory=lambda name, server: FakeMCPClient())
    manager.register_tools()
    storage = Storage(Path(config.output_dir) / "agent.sqlite")

    result, _ = ToolRuntime(Path(config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name="mcp.fetch.fetch",
            args={},
            context=ToolContext("sess", "step", Path(config.output_dir)),
            session_id="sess",
            step_id="step",
            enabled_tools={"mcp.fetch.fetch"},
            permission_rules=[PermissionRule(permission="mcp_server", pattern="*", action="allow"), PermissionRule(permission="mcp_tool", pattern="*", action="allow")],
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "invalid_args"


def test_mcp_tool_permission_denied_before_execution():
    config = _config()
    manager = MCPClientManager(config, client_factory=lambda name, server: FakeMCPClient())
    manager.register_tools()
    storage = Storage(Path(config.output_dir) / "agent.sqlite")

    result, approvals = ToolRuntime(Path(config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name="mcp.fetch.fetch",
            args={"url": "https://example.com"},
            context=ToolContext("sess", "step", Path(config.output_dir)),
            session_id="sess",
            step_id="step",
            enabled_tools={"mcp.fetch.fetch"},
            permission_rules=[PermissionRule(permission="mcp_tool", pattern="*", action="deny")],
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "denied"
    assert approvals[-1]["approved"] is False


def test_mcp_tool_call_writes_tool_events_and_parts():
    config = _config()
    manager = MCPClientManager(config, client_factory=lambda name, server: FakeMCPClient(output="web"))
    manager.register_tools()
    storage = Storage(Path(config.output_dir) / "agent.sqlite")

    result, _ = ToolRuntime(Path(config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name="mcp.fetch.fetch",
            args={"url": "https://example.com"},
            context=ToolContext("sess", "step", Path(config.output_dir)),
            session_id="sess",
            step_id="step",
            enabled_tools={"mcp.fetch.fetch"},
            permission_rules=[PermissionRule(permission="mcp_server", pattern="*", action="allow"), PermissionRule(permission="mcp_tool", pattern="*", action="allow")],
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "success"
    assert "web:fetch" in result.output
    assert storage.list_tool_calls("sess")
    assert any(item["type"] == "tool.completed" for item in storage.list_events("sess"))
    assert storage.list_message_parts("sess")


def test_fetch_mcp_rejects_non_http_url():
    config = _config()
    manager = MCPClientManager(config, client_factory=lambda name, server: FakeMCPClient())
    manager.register_tools()
    storage = Storage(Path(config.output_dir) / "agent.sqlite")

    result, _ = ToolRuntime(Path(config.output_dir) / "artifacts").run(
        ToolRunRequest(
            name="mcp.fetch.fetch",
            args={"url": "file:///etc/passwd"},
            context=ToolContext("sess", "step", Path(config.output_dir)),
            session_id="sess",
            step_id="step",
            enabled_tools={"mcp.fetch.fetch"},
            permission_rules=[PermissionRule(permission="mcp_server", pattern="*", action="allow"), PermissionRule(permission="mcp_tool", pattern="*", action="allow")],
            auto_approve=True,
            storage=storage,
        )
    )

    assert result.status == "invalid_args"
