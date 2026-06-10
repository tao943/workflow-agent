from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.config import AppConfig, PermissionRule
from src.mcp_runtime import MCPClientManager
from src.skills import SkillSpec
from src.tool_providers import BuiltinToolProvider, MCPToolProvider, SkillToolProvider, ToolProvider, ToolProviderStatus


@dataclass
class IntegrationCatalog:
    config: AppConfig
    providers: list[ToolProvider]
    skill_provider: SkillToolProvider
    mcp_provider: MCPToolProvider

    @classmethod
    def build(cls, config: AppConfig) -> "IntegrationCatalog":
        builtin_provider = BuiltinToolProvider()
        skill_provider = SkillToolProvider(config)
        mcp_provider = MCPToolProvider(config)
        providers: list[ToolProvider] = [builtin_provider, skill_provider, mcp_provider]
        return cls(config, providers, skill_provider, mcp_provider)

    @property
    def skill_registry(self) -> dict[str, SkillSpec]:
        return self.skill_provider.registry

    @property
    def skill_catalog(self):
        return self.skill_provider.catalog

    @property
    def mcp_manager(self) -> MCPClientManager:
        return self.mcp_provider.manager

    @property
    def mcp_tool_names(self) -> list[str]:
        return self.mcp_provider.tool_names()

    def discover_all(self, refresh: bool = False):
        tools = []
        for provider in self.providers:
            tools.extend(provider.discover(refresh=refresh))
        return tools

    def register_all(self):
        tools = []
        for provider in self.providers:
            tools.extend(provider.register())
        return tools

    def enabled_tool_names(self, task: str = "", profile: str | None = None, extra_tools: Iterable[str] = ()) -> list[str]:
        enabled = list(dict.fromkeys([*self.skill_provider.tool_names_for_task(task), *self.mcp_provider.tool_names(), *extra_tools]))
        for name, value in self.config.tools.items():
            if value and name not in enabled:
                enabled.append(name)
            if not value and name in enabled:
                enabled.remove(name)
        enabled = self._filter_tools_for_profile(enabled, profile)
        return enabled

    def _filter_tools_for_profile(self, tools: list[str], profile: str | None) -> list[str]:
        if profile != "plan":
            return tools
        write_tools = {"notes_tool", "write_file"}
        return [tool for tool in tools if tool not in write_tools]

    def permission_rules(self, selected_tools: list[str]) -> list[PermissionRule]:
        rules: list[PermissionRule] = []
        for provider in self.providers:
            rules.extend(provider.permission_rules(selected_tools))
        return rules

    def prompt_for_tools(self, selected_tools: list[str]) -> str:
        fragments: list[str] = []
        for provider in self.providers:
            fragments.extend(provider.prompt_fragments(selected_tools))
        return "\n".join(fragment for fragment in fragments if fragment)

    def provider_health(self) -> list[ToolProviderStatus]:
        return [provider.health() for provider in self.providers]

    def provider_by_name(self, name: str) -> ToolProvider | None:
        return next((provider for provider in self.providers if provider.name == name), None)

    def tool_inventory(self, selected_tools: list[str] | None = None) -> list[dict]:
        selected = set(selected_tools or [])
        items = []
        for tool in self.discover_all(refresh=False):
            items.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "provider_name": tool.provider_name,
                    "provider_kind": tool.provider_kind,
                    "source": tool.source,
                    "schema_hash": tool.schema_hash,
                    "trust_level": tool.trust_level,
                    "permissions": tool.permissions,
                    "enabled": not selected or tool.name in selected,
                }
            )
        order = {"builtin": 0, "skill": 1, "mcp": 2}
        return sorted(items, key=lambda item: (order.get(item["provider_kind"], 99), item["name"]))
