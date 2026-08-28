import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
import ipaddress

from pydantic import BaseModel, Field, model_validator


PermissionAction = Literal["allow", "ask", "deny"]


class PermissionRule(BaseModel):
    permission: str
    pattern: str = "*"
    action: PermissionAction = "ask"
    scope: str = "always"
    source: str = "config"


class AgentProfile(BaseModel):
    name: str
    description: str
    skills: list[str] = Field(default_factory=list)
    permissions: list[PermissionRule]


AgentRole = Literal["lead", "member", "reviewer", "researcher", "planner", "builder"]


class AgentSpec(BaseModel):
    name: str
    role: AgentRole
    profile: Literal["build", "plan", "research"] = "build"
    description: str = ""
    system_prompt: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    permissions: list[PermissionRule] = Field(default_factory=list)
    cost_level: Literal["low", "medium", "high"] = "medium"
    can_delegate: bool = False
    can_write: bool = False


class TeamSpec(BaseModel):
    name: str
    description: str
    lead: AgentSpec
    members: list[AgentSpec]
    default_task_policy: str = "lead_assigns_tasks"


class TeamModeConfig(BaseModel):
    enabled: bool = False
    max_parallel_members: int = 4
    max_members: int = 8
    max_messages_per_run: int = 1000
    max_member_turns: int = 50
    message_payload_max_chars: int = 12000
    assignment_strategy: Literal["hybrid", "rules", "llm"] = "hybrid"
    max_assignment_tasks: int = 8
    llm_assignment_enabled: bool = True
    llm_synthesis_enabled: bool = True
    semantic_routing_enabled: bool = True
    min_route_confidence: float = 0.65
    max_replans_per_run: int = 2
    max_repair_tasks: int = 2
    replan_strategy: Literal["after_failure", "after_layer", "before_final"] = "after_layer"


class SkillRuntimeConfig(BaseModel):
    loading_enabled: bool = True
    code_enabled: bool = True
    allow_global_skills: bool = False
    auto_install_dependencies: bool = False
    default_code_action: PermissionAction = "ask"
    allowed_runtimes: list[str] = Field(default_factory=lambda: ["python-subprocess", "docker"])
    paths: list[str] = Field(default_factory=lambda: ["skills"])
    docker_enabled: bool = True
    docker_image: str = "workflow-agent-skill-python:3.13"
    docker_allowed_images: list[str] = Field(default_factory=lambda: ["workflow-agent-skill-python:3.13", "python:3.13-slim"])
    docker_network: str = "none"
    docker_memory: str = "256m"
    docker_cpus: str = "0.5"
    docker_pids_limit: int = 64
    docker_read_only: bool = True
    docker_cap_drop: list[str] = Field(default_factory=lambda: ["ALL"])
    docker_security_opt: list[str] = Field(default_factory=lambda: ["no-new-privileges:true"])


class ContextConfig(BaseModel):
    max_prompt_chars: int = 12000
    recent_turns: int = 3
    tool_output_max_chars: int = 2000
    memory_items: int = 8


class ContextHarnessConfig(BaseModel):
    enabled: bool = True
    offload_threshold_tokens: int = 1200
    max_input_tokens: int = 16000
    reserved_output_tokens: int = 3000
    summary_trigger_ratio: float = 0.75
    retrieval_cache_ttl_seconds: int = 300
    max_retries_per_step: int = 2
    max_repair_branches: int = 2
    stagnation_window: int = 3
    stable_prompt_prefix: bool = True


class MemoryConfig(BaseModel):
    enabled: bool = True
    namespace: str = "project"
    retrieval_limit: int = 8
    candidate_limit: int = 100
    min_active_confidence: float = 0.7
    semantic_rerank_enabled: bool = False
    consolidation_enabled: bool = True


class MCPServerConfig(BaseModel):
    enabled: bool = True
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    allow_tools: list[str] = Field(default_factory=lambda: ["*"])
    deny_tools: list[str] = Field(default_factory=list)
    allow_resources: list[str] = Field(default_factory=list)
    trust_level: Literal["project", "external"] = "external"


class MCPConfig(BaseModel):
    enabled: bool = True
    default_action: PermissionAction = "ask"
    connect_timeout_seconds: int = 10
    tool_timeout_seconds: int = 30
    cache_tool_list_seconds: int = 300
    servers: dict[str, MCPServerConfig] = Field(
        default_factory=lambda: {
            "fetch": MCPServerConfig(
                enabled=True,
                transport="stdio",
                command="uvx",
                args=["mcp-server-fetch"],
                env={"PYTHONIOENCODING": "utf-8"},
                allow_tools=["fetch"],
                trust_level="project",
            ),
            "brave": MCPServerConfig(
                enabled=False,
                transport="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-brave-search"],
                env={"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
                allow_tools=["brave_web_search", "brave_local_search"],
                trust_level="external",
            ),
        }
    )


class EvalConfig(BaseModel):
    evaluation_model: str = "gpt-4o-mini"
    enable_deepeval: bool = True
    enable_external_benchmarks: bool = True
    deepeval_mode: Literal["local", "native"] = "local"
    deep_eval_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "task_completion": 0.7,
            "step_efficiency": 0.5,
            "tool_correctness": 0.8,
            "argument_correctness": 0.8,
            "plan_quality": 0.7,
        }
    )
    benchmark_output_dir: str = "outputs/benchmarks"


class BenchmarkConfig(BaseModel):
    agentbench_path: str = ""
    claw_eval_path: str = ""
    default_trials: int = 3
    allow_resource_heavy: bool = False
    protocol_output_dir: str = "outputs/benchmarks/official"


class A2ARemoteAgentConfig(BaseModel):
    enabled: bool = False
    agent_card_url: str = ""
    token_env: str = ""
    allow_local_fallback: bool = False
    workspace_root: str = "."


class A2AConfig(BaseModel):
    enabled: bool = False
    request_timeout_seconds: int = 30
    task_timeout_seconds: int = 600
    max_retries: int = 2
    artifact_max_bytes: int = 50 * 1024 * 1024
    agent_card_ttl_seconds: int = 300
    allow_insecure_http: bool = False
    agents: dict[str, A2ARemoteAgentConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_endpoint_transport(self):
        for role, agent in self.agents.items():
            if not agent.enabled or not agent.agent_card_url:
                continue
            parsed = urlparse(agent.agent_card_url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(f"A2A {role} endpoint must use HTTP(S)")
            host = (parsed.hostname or "").lower()
            loopback = host in {"localhost", "127.0.0.1", "::1"}
            try:
                loopback = loopback or ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
            if parsed.scheme == "http" and not loopback and not self.allow_insecure_http:
                raise ValueError("non-loopback A2A endpoints require HTTPS")
        return self


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    service_name: str = "workflow-agent"
    jsonl_fallback: str = "outputs/traces/agent.jsonl"
    capture_prompt_content: bool = False


class LangMemConfig(BaseModel):
    enabled: bool = False
    fallback_to_rule_consolidator: bool = True
    max_candidates_per_run: int = 8


class AppConfig(BaseModel):
    model: str = "gpt-4o-mini"
    default_agent: str = "build"
    output_dir: str = "outputs"
    checkpoint_path: str = "outputs/checkpoints/checkpoints.sqlite"
    context: ContextConfig = Field(default_factory=ContextConfig)
    context_harness: ContextHarnessConfig = Field(default_factory=ContextHarnessConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    evals: EvalConfig = Field(default_factory=EvalConfig)
    benchmarks: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    team_mode: TeamModeConfig = Field(default_factory=TeamModeConfig)
    skill_runtime: SkillRuntimeConfig = Field(default_factory=SkillRuntimeConfig)
    a2a: A2AConfig = Field(default_factory=A2AConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    langmem: LangMemConfig = Field(default_factory=LangMemConfig)
    permissions: list[PermissionRule] = Field(default_factory=list)
    skills: dict[str, bool] = Field(default_factory=dict)
    tools: dict[str, bool] = Field(default_factory=dict)


DEFAULT_RULES = [
    PermissionRule(permission="read", pattern="*", action="allow"),
    PermissionRule(permission="write", pattern="*", action="ask"),
    PermissionRule(permission="rag", pattern="*", action="ask"),
    PermissionRule(permission="web", pattern="*", action="ask"),
    PermissionRule(permission="shell", pattern="*", action="deny"),
    PermissionRule(permission="external_directory", pattern="*", action="deny"),
    PermissionRule(permission="team:create", pattern="*", action="ask"),
    PermissionRule(permission="team:message", pattern="*", action="allow"),
    PermissionRule(permission="team:task", pattern="*", action="allow"),
    PermissionRule(permission="team:delete", pattern="*", action="ask"),
    PermissionRule(permission="team:external_worktree", pattern="*", action="deny"),
    PermissionRule(permission="skill", pattern="*", action="ask"),
    PermissionRule(permission="skill_code", pattern="execute", action="ask"),
    PermissionRule(permission="network", pattern="*", action="deny"),
    PermissionRule(permission="dependency_install", pattern="*", action="deny"),
    PermissionRule(permission="mcp_server", pattern="*", action="ask"),
    PermissionRule(permission="mcp_tool", pattern="*", action="ask"),
    PermissionRule(permission="mcp_resource", pattern="*", action="ask"),
]


TEAM_SPECS = {
    "default-dev-team": TeamSpec(
        name="default-dev-team",
        description="A balanced development team with planning, research, building, and review roles.",
        lead=AgentSpec(
            name="coordinator",
            role="lead",
            profile="plan",
            system_prompt="Coordinate the team, split work, collect results, and produce the final answer.",
            allowed_tools=["team_task_create", "team_task_list", "team_task_update", "team_send_message", "team_status"],
            can_delegate=True,
        ),
        members=[
            AgentSpec(name="planner", role="planner", profile="plan", system_prompt="Create precise plans and acceptance criteria."),
            AgentSpec(name="researcher", role="researcher", profile="research", system_prompt="Gather project and knowledge-base context."),
            AgentSpec(name="builder", role="builder", profile="build", system_prompt="Implement safe project changes when requested.", can_write=True),
            AgentSpec(name="reviewer", role="reviewer", profile="plan", system_prompt="Review results, find gaps, and validate completion."),
        ],
    ),
    "hyperplan": TeamSpec(
        name="hyperplan",
        description="A planning-heavy team with multiple reviewers.",
        lead=AgentSpec(name="coordinator", role="lead", profile="plan", system_prompt="Coordinate planning work.", can_delegate=True),
        members=[
            AgentSpec(name="planner", role="planner", profile="plan", system_prompt="Draft the main plan."),
            AgentSpec(name="architecture-reviewer", role="reviewer", profile="plan", system_prompt="Review architecture risks."),
            AgentSpec(name="test-reviewer", role="reviewer", profile="plan", system_prompt="Review test strategy."),
            AgentSpec(name="product-reviewer", role="reviewer", profile="plan", system_prompt="Review user-facing value and scope."),
        ],
    ),
    "research-build": TeamSpec(
        name="research-build",
        description="A team that researches first, builds second, and reviews last.",
        lead=AgentSpec(name="coordinator", role="lead", profile="plan", system_prompt="Sequence research, build, and review work.", can_delegate=True),
        members=[
            AgentSpec(name="researcher", role="researcher", profile="research", system_prompt="Collect relevant context before execution."),
            AgentSpec(name="builder", role="builder", profile="build", system_prompt="Use research findings to execute the task.", can_write=True),
            AgentSpec(name="reviewer", role="reviewer", profile="plan", system_prompt="Validate results and list remaining risks."),
        ],
    ),
}


AGENT_PROFILES = {
    "build": AgentProfile(
        name="build",
        description="默认构建模式，可执行写文件等操作，但危险动作需要确认。",
        skills=["core", "notes", "file_ops", "research"],
        permissions=DEFAULT_RULES,
    ),
    "plan": AgentProfile(
        name="plan",
        description="只读规划模式，不允许写文件、shell、web 或 RAG 写入类动作。",
        skills=["core", "file_ops", "research"],
        permissions=[
            PermissionRule(permission="read", pattern="*", action="allow"),
            PermissionRule(permission="write", pattern="*", action="deny"),
            PermissionRule(permission="rag", pattern="*", action="ask"),
            PermissionRule(permission="web", pattern="*", action="deny"),
            PermissionRule(permission="shell", pattern="*", action="deny"),
            PermissionRule(permission="external_directory", pattern="*", action="deny"),
            PermissionRule(permission="mcp_server", pattern="*", action="deny"),
            PermissionRule(permission="mcp_tool", pattern="*", action="deny"),
            PermissionRule(permission="mcp_resource", pattern="*", action="deny"),
        ],
    ),
    "research": AgentProfile(
        name="research",
        description="研究模式，允许读取和 RAG 检索，写文件仍需确认。",
        skills=["core", "file_ops", "research", "notes"],
        permissions=[
            PermissionRule(permission="read", pattern="*", action="allow"),
            PermissionRule(permission="write", pattern="*", action="ask"),
            PermissionRule(permission="rag", pattern="*", action="allow"),
            PermissionRule(permission="web", pattern="*", action="ask"),
            PermissionRule(permission="shell", pattern="*", action="deny"),
            PermissionRule(permission="external_directory", pattern="*", action="deny"),
            PermissionRule(permission="mcp_server", pattern="*", action="ask"),
            PermissionRule(permission="mcp_tool", pattern="*", action="ask"),
            PermissionRule(permission="mcp_resource", pattern="*", action="ask"),
        ],
    ),
}


def load_config(path: str | None = None, agent: str | None = None) -> AppConfig:
    config_path = Path(path or "agent_config.json")
    if config_path.exists():
        config = AppConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    else:
        config = AppConfig(permissions=DEFAULT_RULES)

    if not agent:
        return config

    return config.model_copy(update={"default_agent": agent})


def permissions_for_agent(config: AppConfig) -> list[PermissionRule]:
    profile = AGENT_PROFILES.get(config.default_agent)
    if not profile:
        return config.permissions
    return profile.permissions + config.permissions


def skills_for_agent(config: AppConfig) -> list[str]:
    profile = AGENT_PROFILES.get(config.default_agent)
    skill_names = list(profile.skills if profile else [])
    for name, enabled in config.skills.items():
        if enabled and name not in skill_names:
            skill_names.append(name)
        if not enabled and name in skill_names:
            skill_names.remove(name)
    return skill_names


def get_team_spec(name: str) -> TeamSpec:
    if name not in TEAM_SPECS:
        raise ValueError(f"Unknown team: {name}")
    return TEAM_SPECS[name]
