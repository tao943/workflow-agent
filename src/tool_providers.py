from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from src.config import AppConfig, PermissionRule, SkillRuntimeConfig
from src.mcp_runtime import MCPClientManager, MCPToolInfo, load_and_register_mcp_tools
from src.skills import (
    SkillCatalog,
    SkillSpec,
    build_skill_registry,
    load_and_register_project_skills,
    permissions_for_skills,
    prompt_for_skills,
    select_skills_for_task,
    tools_for_skills,
)
from src.tools.registry import TOOL_REGISTRY, ToolSpec, register_tools


@dataclass(frozen=True)
class ToolProviderStatus:
    name: str
    kind: str
    ok: bool
    enabled: bool = True
    tools: list[str] | None = None
    error: str = ""


class ToolProvider(Protocol):
    name: str
    kind: str

    def discover(self, refresh: bool = False) -> list[ToolSpec]: ...

    def register(self) -> list[ToolSpec]: ...

    def tool_names(self) -> list[str]: ...

    def permission_rules(self, selected: list[str] | None = None) -> list[PermissionRule]: ...

    def prompt_fragments(self, selected: list[str] | None = None) -> list[str]: ...

    def health(self) -> ToolProviderStatus: ...


class BuiltinToolProvider:
    name = "builtin"
    kind = "builtin"

    def discover(self, refresh: bool = False) -> list[ToolSpec]:
        return [
            self._with_metadata(tool)
            for _, tool in sorted(TOOL_REGISTRY.items())
            if not tool.name.startswith("mcp.") and tool.provider_kind == self.kind and tool.source == "builtin"
        ]

    def register(self) -> list[ToolSpec]:
        return self.discover()

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.discover()]

    def permission_rules(self, selected: list[str] | None = None) -> list[PermissionRule]:
        return []

    def prompt_fragments(self, selected: list[str] | None = None) -> list[str]:
        names = selected or self.tool_names()
        return [f"- builtin: {', '.join(names)}"] if names else []

    def health(self) -> ToolProviderStatus:
        names = self.tool_names()
        return ToolProviderStatus(self.name, self.kind, True, True, names)

    def _with_metadata(self, tool: ToolSpec) -> ToolSpec:
        if tool.provider_name == self.name and tool.provider_kind == self.kind:
            return tool
        return replace(tool, provider_name=self.name, provider_kind=self.kind, source="builtin", trust_level="project")


class SkillToolProvider:
    name = "skill"
    kind = "skill"

    def __init__(self, config: AppConfig, catalog: SkillCatalog | None = None, registry: dict[str, SkillSpec] | None = None) -> None:
        self.config = config
        self.catalog = catalog or load_and_register_project_skills(config)
        self.registry = registry or build_skill_registry(self.catalog, config.skill_runtime)

    def discover(self, refresh: bool = False) -> list[ToolSpec]:
        return [tool for tool in TOOL_REGISTRY.values() if tool.name in self.catalog.dynamic_tool_names()]

    def register(self) -> list[ToolSpec]:
        if self.config.skill_runtime.code_enabled:
            for package in self.catalog.packages.values():
                from src.skill_plugins import package_to_tool_specs

                specs = [
                    replace(
                        tool,
                        provider_name=self.name,
                        provider_kind=self.kind,
                        source=f"skill:{package.name}",
                        trust_level="project",
                    )
                    for tool in package_to_tool_specs(package, self.config.skill_runtime)
                ]
                register_tools(specs, replace=True)
        return self.discover()

    def select_skills(self, task: str) -> list[str]:
        from src.config import skills_for_agent

        return select_skills_for_task(skills_for_agent(self.config), task, self.registry, self.config)

    def tool_names(self) -> list[str]:
        return sorted(self.catalog.dynamic_tool_names())

    def tool_names_for_task(self, task: str) -> list[str]:
        return tools_for_skills(self.select_skills(task), self.registry)

    def permission_rules(self, selected: list[str] | None = None) -> list[PermissionRule]:
        if not selected:
            return []
        skill_names = [name for name, skill in self.registry.items() if any(tool in selected for tool in skill.tools)]
        return permissions_for_skills(skill_names, self.registry)

    def prompt_fragments(self, selected: list[str] | None = None) -> list[str]:
        if not selected:
            return []
        skill_names = [name for name, skill in self.registry.items() if any(tool in selected for tool in skill.tools)]
        prompt = prompt_for_skills(skill_names, self.registry)
        return [prompt] if prompt else []

    def health(self) -> ToolProviderStatus:
        errors = [error.message for error in self.catalog.errors]
        return ToolProviderStatus(self.name, self.kind, not errors, True, self.tool_names(), "; ".join(errors))

    def _owns_tool(self, tool_name: str) -> bool:
        return tool_name in self.catalog.dynamic_tool_names()


class MCPToolProvider:
    name = "mcp"
    kind = "mcp"

    def __init__(self, config: AppConfig, manager: MCPClientManager | None = None) -> None:
        self.config = config
        self.manager = manager or load_and_register_mcp_tools(config)

    def discover(self, refresh: bool = False) -> list[ToolSpec]:
        infos = self.manager.discover_tools(refresh=True) if refresh else self.manager.configured_tool_infos()
        return [self._tool_spec(info) for info in infos]

    def register(self) -> list[ToolSpec]:
        infos = self.manager.register_tools(replace=True)
        return [self._tool_spec(info) for info in infos]

    def tool_names(self) -> list[str]:
        return [item.local_name for item in self.manager.configured_tool_infos()]

    def permission_rules(self, selected: list[str] | None = None) -> list[PermissionRule]:
        return []

    def prompt_fragments(self, selected: list[str] | None = None) -> list[str]:
        names = selected or self.tool_names()
        return [f"- mcp: {', '.join(names)}"] if names else []

    def health(self) -> ToolProviderStatus:
        tools = self.tool_names()
        return ToolProviderStatus(self.name, self.kind, True, self.config.mcp.enabled, tools)

    def _tool_spec(self, info: MCPToolInfo) -> ToolSpec:
        tool = self.manager._tool_spec(info)
        return replace(
            tool,
            provider_name=self.name,
            provider_kind=self.kind,
            source=f"mcp:{info.server}",
            schema_hash=info.schema_hash,
            trust_level=info.trust_level,
        )
