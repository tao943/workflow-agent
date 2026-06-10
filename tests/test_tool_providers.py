from pathlib import Path

from src.config import AppConfig
from src.integrations import IntegrationCatalog
from src.mcp_runtime import MCPClientManager
from src.skill_plugins import load_skill_catalog
from src.skills import build_skill_registry
from src.tool_providers import BuiltinToolProvider, MCPToolProvider, SkillToolProvider


def test_builtin_tool_provider_lists_core_tools() -> None:
    provider = BuiltinToolProvider()

    names = provider.tool_names()

    assert "calculator" in names
    assert "datetime_tool" in names
    assert "list_files" in names
    assert provider.health().ok is True


def test_skill_tool_provider_discovers_project_skill() -> None:
    config = AppConfig()
    catalog = load_skill_catalog(config.skill_runtime, workspace=Path.cwd())
    provider = SkillToolProvider(config, catalog=catalog, registry=build_skill_registry(catalog, config.skill_runtime))
    provider.register()

    names = provider.tool_names()

    assert "demo_echo.echo" in names
    assert any("skill_code" in rule.permission for rule in provider.permission_rules(["demo_echo.echo"]))
    assert provider.prompt_fragments(["demo_echo.echo"])


def test_mcp_tool_provider_uses_configured_static_tools() -> None:
    config = AppConfig()
    manager = MCPClientManager(config)
    provider = MCPToolProvider(config, manager)

    names = provider.tool_names()
    tools = provider.discover(refresh=False)

    assert "mcp.fetch.fetch" in names
    assert tools[0].provider_kind == "mcp"
    assert tools[0].schema_hash


def test_integration_catalog_selects_enabled_tools_with_stable_order() -> None:
    config = AppConfig(tools={"datetime_tool": False, "extra_tool": True})
    catalog = IntegrationCatalog.build(config)
    catalog.register_all()

    selected = catalog.enabled_tool_names("请计算 1+1", config.default_agent)
    inventory = catalog.tool_inventory(selected)

    assert "calculator" in selected
    assert "datetime_tool" not in selected
    assert "mcp.fetch.fetch" in selected
    assert "extra_tool" in selected
    assert [item["provider_kind"] for item in inventory][:2] == ["builtin", "builtin"]


def test_integration_catalog_hides_write_tools_for_plan_profile() -> None:
    config = AppConfig(tools={"notes_tool": True, "write_file": True, "read_file": True})
    catalog = IntegrationCatalog.build(config)
    catalog.register_all()

    selected = catalog.enabled_tool_names("帮我制定计划并保存为笔记", "plan")

    assert "notes_tool" not in selected
    assert "write_file" not in selected
    assert "read_file" in selected
