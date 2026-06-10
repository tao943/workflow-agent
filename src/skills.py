from dataclasses import dataclass, field
from pathlib import Path

from src.config import AppConfig, PermissionRule, SkillRuntimeConfig
from src.skill_plugins import SkillCatalog, SkillPackage, load_skill_catalog, package_permission_rules, package_to_tool_specs
from src.tools.registry import register_tools


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    permissions: list[PermissionRule]
    examples: list[str] = field(default_factory=list)
    source: str = "builtin"
    path: str | None = None
    triggers: list[str] = field(default_factory=list)
    enabled_by_default: bool = False
    metadata: dict = field(default_factory=dict)


SKILL_REGISTRY = {
    "core": SkillSpec(
        name="core",
        description="基础任务处理能力，包括计算和时间查询。",
        system_prompt="你可以使用基础工具完成数学计算、日期时间查询和简单整理。",
        tools=["calculator", "datetime_tool"],
        permissions=[],
        examples=["计算 12 * (8 + 5)", "告诉我今天日期"],
        triggers=["计算", "时间", "日期"],
        enabled_by_default=True,
    ),
    "notes": SkillSpec(
        name="notes",
        description="笔记保存能力，用于把结果保存为项目内 Markdown 笔记。",
        system_prompt="当用户明确要求保存、记录或生成笔记时，可以规划 notes_tool。",
        tools=["notes_tool"],
        permissions=[PermissionRule(permission="write", pattern="outputs/notes.md", action="ask")],
        examples=["保存为笔记", "记录当前计划"],
        triggers=["保存", "笔记", "记录"],
    ),
    "file_ops": SkillSpec(
        name="file_ops",
        description="项目内文件读写和搜索能力。",
        system_prompt="你可以读取、列出、搜索项目内文件；只有在构建模式允许时才写文件。",
        tools=["read_file", "write_file", "list_files", "search_files"],
        permissions=[
            PermissionRule(permission="read", pattern="*", action="allow"),
            PermissionRule(permission="write", pattern="*", action="ask"),
        ],
        examples=["读取 README.md", "列出项目文件", "搜索 LangGraph"],
        triggers=["项目", "文件", "源码", "结构", "搜索"],
    ),
    "research": SkillSpec(
        name="research",
        description="RAG 知识库检索能力。",
        system_prompt="当任务需要查知识库、检索资料或基于 RAG 回答时，优先规划 rag_search。",
        tools=["rag_search"],
        permissions=[PermissionRule(permission="rag", pattern="*", action="ask")],
        examples=["检索知识库里的 LangGraph checkpoint"],
        triggers=["RAG", "知识库", "检索", "资料", "checkpoint"],
    ),
}


def build_skill_registry(catalog: SkillCatalog | None = None, runtime_config: SkillRuntimeConfig | None = None) -> dict[str, SkillSpec]:
    registry = dict(SKILL_REGISTRY)
    runtime_config = runtime_config or SkillRuntimeConfig()
    if catalog:
        for package in catalog.packages.values():
            registry[package.name] = skill_spec_from_package(package, runtime_config)
    return registry


def skill_spec_from_package(package: SkillPackage, runtime_config: SkillRuntimeConfig | None = None) -> SkillSpec:
    action = (runtime_config or SkillRuntimeConfig()).default_code_action
    return SkillSpec(
        name=package.name,
        description=package.description,
        system_prompt=package.system_prompt,
        tools=[tool.name for tool in package.tools],
        permissions=package_permission_rules(package, action=action),
        examples=[],
        source="project",
        path=str(package.root),
        triggers=package.triggers,
        metadata={
            "version": package.version,
            "runtime": package.runtime,
            "image": package.image,
            "manifest_path": str(package.manifest_path),
            "entrypoint": str(package.entrypoint),
            "network": package.network,
            "dependencies": package.dependencies,
            "filesystem": package.filesystem,
        },
    )


def load_and_register_project_skills(config: AppConfig, workspace: Path | None = None) -> SkillCatalog:
    catalog = load_skill_catalog(config.skill_runtime, workspace=workspace)
    if config.skill_runtime.code_enabled:
        for package in catalog.packages.values():
            register_tools(package_to_tool_specs(package, config.skill_runtime), replace=True)
    return catalog


def select_skills_for_task(base_names: list[str], task: str, registry: dict[str, SkillSpec], config: AppConfig) -> list[str]:
    selected = list(base_names)
    lowered = task.lower()
    for name, skill in registry.items():
        if name in selected:
            continue
        if config.skills.get(name) is False:
            continue
        if config.skills.get(name) is True or any(trigger.lower() in lowered for trigger in skill.triggers):
            selected.append(name)
    return selected


def skills_for_names(names: list[str], registry: dict[str, SkillSpec] | None = None) -> list[SkillSpec]:
    source = registry or SKILL_REGISTRY
    return [source[name] for name in names if name in source]


def tools_for_skills(skill_names: list[str], registry: dict[str, SkillSpec] | None = None) -> list[str]:
    tools: list[str] = []
    for skill in skills_for_names(skill_names, registry):
        for tool in skill.tools:
            if tool not in tools:
                tools.append(tool)
    return tools


def permissions_for_skills(skill_names: list[str], registry: dict[str, SkillSpec] | None = None) -> list[PermissionRule]:
    permissions: list[PermissionRule] = []
    for skill in skills_for_names(skill_names, registry):
        permissions.extend(skill.permissions)
    return permissions


def prompt_for_skills(skill_names: list[str], registry: dict[str, SkillSpec] | None = None) -> str:
    lines: list[str] = []
    for skill in skills_for_names(skill_names, registry):
        lines.append(f"- {skill.name}: {skill.description} {skill.system_prompt}")
    return "\n".join(lines)
