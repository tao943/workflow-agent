# A2A, LangMem, and OpenInference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently deployable A2A Researcher, Builder, and Reviewer HTTP services, standard OpenInference/OpenTelemetry tracing, and LangMem-backed candidate extraction without bypassing the existing Team DAG, ToolRuntime, Evidence Gate, or MemoryPolicyGate.

**Architecture:** Keep `TeamRuntime` as the single orchestrator and introduce a project-owned `MemberExecutor` boundary with local and A2A implementations. Hide the official A2A SDK, LangMem, and OpenTelemetry/OpenInference APIs behind focused adapters so local/offline behavior remains available and external failures degrade according to the approved role policy.

**Tech Stack:** Python 3.13, LangGraph 1.x, `a2a-sdk[fastapi]` 1.x, FastAPI/Uvicorn, LangMem 0.0.x, OpenTelemetry 1.x, OpenInference LangChain instrumentation 0.1.x, HTTPX, Pydantic 2, SQLite, pytest.

## Global Constraints

- Keep `TeamRuntime` as the only task-DAG owner; remote agents cannot edit the DAG, permission rules, or system instructions.
- Deploy Researcher, Builder, and Reviewer as independent HTTP processes with independent absolute workspace roots; file tools never depend on process CWD.
- Require HTTPS for non-loopback A2A endpoints unless the default-off development override `allow_insecure_http` is explicit.
- Keep business attempts separate from transport retries and propagate W3C Trace Context across every A2A HTTP boundary.
- Read Bearer tokens only from environment variables and redact them from events, traces, errors, memory, and Agent Cards.
- Permit local fallback only for Researcher and Reviewer; Builder failures must remain blocking.
- Inline small results; validate origin, redirect behavior, size, MIME, and SHA-256 before accepting remote artifacts.
- Keep `MemoryGovernanceRuntime` and `MemoryPolicyGate` as the only long-term memory write path.
- Treat LangMem extraction and OTLP export as non-critical: fall back to rule consolidation and JSONL tracing.
- Do not add Mem0, Graphiti, Neo4j, Phoenix, SWE-ReX, E2B, or another agent framework.
- Keep existing local, offline, MCP, Skill, Eval, and Team behavior compatible when new features are disabled.
- Use test-first Red-Green-Refactor for every production behavior.
- Do not stage or commit the user's unrelated `.gitignore`, Husky, Prettier, package, evaluation document, or `tmp/` changes.

---

## File Map

### New production files

- `src/observability.py`: trace interfaces, secret redaction, JSONL fallback, and optional OTLP/OpenInference initialization.
- `src/member_execution.py`: local/remote-neutral member request/result contracts and the local executor.
- `src/a2a_runtime.py`: A2A registry, SDK client adapter, state normalization, retries, recovery IDs, and artifact download validation.
- `src/a2a_service.py`: role Agent Cards, Bearer authentication, A2A `AgentExecutor`, protected artifact endpoint, and FastAPI application factory.
- `src/langmem_runtime.py`: structured LangMem extraction and conversion to governed `MemoryCandidate` objects.

### Modified production files

- `requirements.txt`: bounded dependencies, including a temporary MCP v1 upper bound.
- `src/config.py`: A2A, observability, and LangMem Pydantic configuration.
- `agent_config.json.example`: documented role endpoints and feature toggles.
- `.env.example`: token and OTLP environment variable names only.
- `src/storage.py`: persistent A2A task mappings and service artifact metadata.
- `src/team.py`: executor routing, remote recovery/fallback, tracing, and centralized LangMem candidate governance.
- `src/runtime.py`: root agent spans and shared TraceProvider wiring.
- `src/tool_runtime.py`: tool spans with safe attributes.
- `src/mcp_runtime.py`: MCP spans with safe attributes.
- `src/context_harness.py`: memory/context span boundaries.
- `src/main.py`: A2A service, health, inventory, direct-role, and observability doctor commands.
- `README.md`: deployment and operator documentation.
- `pytest.ini`: `a2a` and `observability` markers for optional real tests.

### New and modified tests

- `tests/test_observability.py`
- `tests/test_member_execution.py`
- `tests/test_a2a_runtime.py`
- `tests/test_a2a_service.py`
- `tests/test_team_a2a.py`
- `tests/test_langmem_runtime.py`
- `tests/test_a2a_real.py`
- `tests/test_main_config.py`
- `tests/test_runtime.py`
- `tests/test_tool_runtime.py`
- `tests/test_mcp_runtime.py`

---

### Task 1: Bound dependencies and add configuration contracts

**Files:**

- Modify: `requirements.txt`
- Modify: `src/config.py:51-188`
- Modify: `agent_config.json.example`
- Modify: `.env.example`
- Modify: `tests/test_main_config.py`

**Interfaces:**

- Produces: `A2ARemoteAgentConfig`, `A2AConfig`, `ObservabilityConfig`, and `LangMemConfig` Pydantic models.
- Produces: `AppConfig.a2a`, `AppConfig.observability`, and `AppConfig.langmem`.
- Consumes later: Tasks 2-9 use these exact configuration names.

- [ ] **Step 1: Write failing configuration tests**

Add tests that define the public configuration contract:

```python
from src.config import A2AConfig, A2ARemoteAgentConfig, AppConfig


def test_a2a_config_uses_safe_role_defaults():
    config = A2AConfig(
        enabled=True,
        agents={
            "builder": A2ARemoteAgentConfig(
                enabled=True,
                agent_card_url="http://127.0.0.1:8102/.well-known/agent-card.json",
                token_env="A2A_BUILDER_TOKEN",
                allow_local_fallback=False,
            )
        },
    )
    assert config.max_retries == 2
    assert config.artifact_max_bytes == 50 * 1024 * 1024
    assert config.agents["builder"].allow_local_fallback is False


def test_app_config_disables_external_features_by_default():
    config = AppConfig()
    assert config.a2a.enabled is False
    assert config.a2a.allow_insecure_http is False
    assert config.observability.capture_prompt_content is False
    assert config.langmem.enabled is False


def test_non_loopback_http_requires_explicit_development_override():
    with pytest.raises(ValueError, match="HTTPS"):
        A2AConfig(
            enabled=True,
            agents={
                "builder": A2ARemoteAgentConfig(
                    enabled=True,
                    agent_card_url="http://10.0.0.8:8102/.well-known/agent-card.json",
                    token_env="A2A_BUILDER_TOKEN",
                )
            },
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_main_config.py -q
```

Expected: FAIL because the new configuration classes and `AppConfig` fields do not exist.

- [ ] **Step 3: Add minimal Pydantic configuration models**

Add these exact public fields to `src/config.py`:

```python
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


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    service_name: str = "workflow-agent"
    jsonl_fallback: str = "outputs/traces/agent.jsonl"
    capture_prompt_content: bool = False


class LangMemConfig(BaseModel):
    enabled: bool = False
    fallback_to_rule_consolidator: bool = True
    max_candidates_per_run: int = 8
```

Add `a2a`, `observability`, and `langmem` fields to `AppConfig` using `Field(default_factory=...)`. Validate endpoint transport centrally: plain HTTP is accepted only for loopback hosts; any non-loopback HTTP URL is rejected unless the global development-only `allow_insecure_http` flag is explicitly true. The flag defaults false and must never be inferred from host binding.

- [ ] **Step 4: Bound and add dependencies**

Update `requirements.txt` with explicit compatibility ranges:

```text
langgraph>=1.2,<2
langchain-openai>=0.3,<2
python-dotenv>=1.0.1,<2
pytest>=8.3,<10
pydantic>=2.11,<3
mcp>=1.27,<2
jsonschema>=4.22,<5
a2a-sdk[fastapi]>=1.1,<2
uvicorn>=0.35,<1
httpx>=0.28,<1
langmem>=0.0.30,<0.1
opentelemetry-api>=1.44,<2
opentelemetry-sdk>=1.44,<2
opentelemetry-exporter-otlp-proto-http>=1.44,<2
openinference-instrumentation-langchain>=0.1.73,<0.2
```

The `mcp<2` bound is mandatory until `src/mcp_runtime.py` is migrated to MCP SDK v2 in a separate change.

- [ ] **Step 5: Update example configuration without secrets**

Add the approved `a2a`, `observability`, and `langmem` blocks to `agent_config.json.example`. Add only these names to `.env.example`:

```text
A2A_RESEARCHER_TOKEN=
A2A_BUILDER_TOKEN=
A2A_REVIEWER_TOKEN=
OTEL_EXPORTER_OTLP_ENDPOINT=
```

- [ ] **Step 6: Install dependencies and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests\test_main_config.py -q
```

Expected: dependency installation exits 0 and the configuration tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add requirements.txt src/config.py agent_config.json.example .env.example tests/test_main_config.py
git commit -m "feat: configure A2A memory and observability"
```

---

### Task 2: Add safe tracing with JSONL and optional OTLP/OpenInference

**Files:**

- Create: `src/observability.py`
- Create: `tests/test_observability.py`
- Modify: `src/runtime.py:48-342`
- Modify: `tests/test_runtime.py`

**Interfaces:**

- Produces: `TraceProvider.start_span(name: str, attributes: dict[str, Any] | None = None)` context manager.
- Produces: `TraceSpan.set_attribute`, `TraceSpan.record_exception`, and `TraceSpan.identifiers`.
- Produces: `TraceProvider.inject(carrier)` and `TraceProvider.extract(carrier)` using the W3C Trace Context propagator.
- Produces: `build_trace_provider(config: ObservabilityConfig, output_dir: str) -> TraceProvider`.
- Produces: `redact_sensitive(value: Any) -> Any`.
- Consumes later: Tool, MCP, Team, A2A, Artifact, and LangMem integrations.

- [ ] **Step 1: Write failing JSONL and redaction tests**

```python
import json

from src.observability import JsonlTraceProvider, redact_sensitive


def test_redact_sensitive_recursively_removes_credentials():
    value = {
        "authorization": "Bearer secret-token",
        "nested": {"api_key": "sk-secret", "safe": "kept"},
    }
    assert redact_sensitive(value) == {
        "authorization": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]", "safe": "kept"},
    }


def test_jsonl_provider_writes_parent_child_spans(tmp_path):
    path = tmp_path / "trace.jsonl"
    provider = JsonlTraceProvider(path, service_name="workflow-agent-test")
    with provider.start_span("agent.run", {"token": "secret"}) as root:
        with provider.start_span("team.plan", {"team": "default-dev-team"}) as child:
            assert child.identifiers.trace_id == root.identifiers.trace_id
            assert child.identifiers.parent_span_id == root.identifiers.span_id
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["name"] for item in records] == ["team.plan", "agent.run"]
    assert records[-1]["attributes"]["token"] == "[REDACTED]"


def test_w3c_context_round_trips_between_providers():
    lead_provider, remote_provider = _linked_otel_providers()
    carrier = {}
    with lead_provider.start_span("lead") as lead:
        lead_provider.inject(carrier)
    with remote_provider.extract(carrier):
        with remote_provider.start_span("remote") as remote:
            assert remote.identifiers.trace_id == lead.identifiers.trace_id
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_observability.py -q
```

Expected: FAIL because `src.observability` does not exist.

- [ ] **Step 3: Implement the provider boundary**

Create these concrete types in `src/observability.py`:

```python
@dataclass(frozen=True)
class TraceIdentifiers:
    trace_id: str
    span_id: str
    parent_span_id: str | None = None


class TraceSpan(Protocol):
    identifiers: TraceIdentifiers
    def set_attribute(self, key: str, value: Any) -> None: ...
    def record_exception(self, exc: BaseException) -> None: ...


class TraceProvider(Protocol):
    @contextmanager
    def start_span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Iterator[TraceSpan]: ...
    def inject(self, carrier: MutableMapping[str, str]) -> None: ...
    @contextmanager
    def extract(self, carrier: Mapping[str, str]) -> Iterator[None]: ...
```

Implement `NoOpTraceProvider`, thread-safe `JsonlTraceProvider`, and `OpenTelemetryTraceProvider`. Use a `ContextVar` stack so spans created in one member thread retain correct parents without leaking to other Team threads. Use OpenTelemetry's W3C `TraceContextTextMapPropagator` for carrier injection/extraction; preserve `tracestate`, ignore malformed remote context, and never log raw propagation headers.

`OpenTelemetryTraceProvider` must configure `OTLPSpanExporter`, `BatchSpanProcessor`, and OpenInference `LangChainInstrumentor`. Its exporter failure path must write a `trace_export_error` JSONL record instead of raising into Runtime code.

- [ ] **Step 4: Wire root Runtime spans**

Extend `AgentRuntime.__init__` with an optional `trace_provider` argument and default it through `build_trace_provider`. Wrap `AgentRuntime.run` with:

```python
with self.trace_provider.start_span(
    "agent.run",
    {
        "agent.profile": self.config.default_agent,
        "agent.team": options.team or "",
        "session.id": options.session_id or "",
    },
) as run_span:
    return self._run_traced(options, run_span)
```

Move the existing method body into `_run_traced` without changing behavior.

- [ ] **Step 5: Verify GREEN and regression compatibility**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_observability.py tests\test_runtime.py -q
```

Expected: all selected tests pass and no trace export configuration is required.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/observability.py src/runtime.py tests/test_observability.py tests/test_runtime.py
git commit -m "feat: add OpenInference trace provider"
```

---

### Task 3: Introduce the MemberExecutor boundary without changing local behavior

**Files:**

- Create: `src/member_execution.py`
- Create: `tests/test_member_execution.py`
- Modify: `src/runtime.py`
- Modify: `src/tool_runtime.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_tool_runtime.py`
- Modify: `src/team.py:313-566`
- Modify: `tests/test_team_mode.py`

**Interfaces:**

- Produces: immutable `MemberTaskRequest` and mutable `MemberExecutionResult` dataclasses.
- Produces: `MemberExecutor.execute(request: MemberTaskRequest) -> MemberExecutionResult`.
- Produces: `LocalMemberExecutor` wrapping the existing member `AgentRuntime` behavior.
- Consumes later: `A2AMemberExecutor` implements the same protocol in Task 5.

- [ ] **Step 1: Write a failing local-executor contract test**

```python
from src.member_execution import LocalMemberExecutor, MemberTaskRequest


def test_local_member_executor_preserves_runtime_result(fake_agent_runtime):
    executor = LocalMemberExecutor(lambda config: fake_agent_runtime)
    request = MemberTaskRequest(
        team_run_id="team_run_1",
        task_id="team_task_1",
        logical_task_id="collect_project_evidence",
        member_name="researcher",
        role="researcher",
        profile="research",
        title="Collect evidence",
        description="Read project evidence",
        required_tools=["list_files"],
        required_evidence=["file list"],
        acceptance_criteria="Tool evidence exists.",
        session_id="session_researcher",
        auto_approve=False,
        output_format="default",
        no_memory=True,
        workspace_root=str(tmp_path.resolve()),
    )
    result = executor.execute(request)
    assert result.status == "completed"
    assert result.member_name == "researcher"
    assert result.execution_mode == "local"
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_member_execution.py -q
```

Expected: FAIL because `src.member_execution` does not exist.

- [ ] **Step 3: Define the stable contracts**

The request must expose only serializable task data and not the mutable `AgentSpec` object:

```python
@dataclass(frozen=True)
class MemberTaskRequest:
    team_run_id: str
    task_id: str
    logical_task_id: str
    member_name: str
    role: str
    profile: str
    title: str
    description: str
    required_tools: list[str]
    required_evidence: list[str]
    acceptance_criteria: str
    session_id: str
    auto_approve: bool
    output_format: str
    no_memory: bool
    workspace_root: str
    attempt: int = 0


@dataclass
class MemberExecutionResult:
    member_name: str
    role: str
    status: str
    final_answer: str
    acceptance: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    events: list[dict[str, Any]]
    execution_mode: str
    error: str = ""
    remote_context_id: str = ""
    remote_task_id: str = ""
    trace: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Extract existing local execution into LocalMemberExecutor**

Move the member-specific `AgentRuntime.run(RuntimeOptions(...))` call from `TeamRuntime._run_member_task` into `LocalMemberExecutor`. Keep task persistence, mailbox messages, Evidence Gate, and member report construction in `TeamRuntime`; they remain Lead responsibilities.

Resolve `workspace_root` to an absolute path at configuration/service startup and propagate the same value through `MemberTaskRequest` -> `RuntimeOptions` -> Agent graph state -> `ToolContext`. All file tools must resolve requested paths against this root, reject traversal/symlink escapes, and never consult process CWD. Add a contract test with three distinct role roots proving each role observes only its own files and that Builder writes cannot escape its root.

Inject `member_executor_factory` into `TeamRuntime.__init__` so tests can supply deterministic executors.

- [ ] **Step 5: Verify local Team behavior is unchanged**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_member_execution.py tests\test_team_mode.py tests\test_runtime.py tests\test_tool_runtime.py -q
```

Expected: all selected tests pass, including existing local Team tests.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/member_execution.py src/runtime.py src/tool_runtime.py src/team.py tests/test_member_execution.py tests/test_team_mode.py tests/test_runtime.py tests/test_tool_runtime.py
git commit -m "refactor: isolate team member execution"
```

---

### Task 4: Persist A2A mappings and validate registry/security metadata

**Files:**

- Modify: `src/storage.py:33-225`
- Create: `src/a2a_runtime.py`
- Create: `tests/test_a2a_runtime.py`

**Interfaces:**

- Produces storage methods `upsert_a2a_task`, `get_a2a_task`, `add_a2a_artifact`, and `get_a2a_artifact`.
- Produces `A2AAgentRegistry.resolve(role: str) -> ResolvedA2AAgent`.
- Produces `A2ARuntimeError` with stable `code`, `retryable`, and `details` fields.
- Consumes later: A2A client, service, Team routing, health CLI.

- [ ] **Step 1: Write failing storage and registry tests**

```python
def test_storage_round_trips_a2a_task_mapping(tmp_path):
    storage = Storage(tmp_path / "agent.sqlite")
    storage.upsert_a2a_task(
        team_run_id="run_1",
        logical_task_id="review",
        role="reviewer",
        business_attempt=0,
        execution_id="exec_review_1",
        idempotency_key="run_1:review:0",
        transport_retry_count=1,
        context_id="ctx_1",
        remote_task_id="remote_1",
        endpoint="http://127.0.0.1:8103",
        status="working",
    )
    row = storage.get_a2a_task_by_execution_id("exec_review_1")
    assert row["remote_task_id"] == "remote_1"
    assert row["transport_retry_count"] == 1


def test_registry_reads_token_without_exposing_it(monkeypatch):
    monkeypatch.setenv("A2A_RESEARCHER_TOKEN", "secret-token")
    registry = A2AAgentRegistry(_config_for("researcher"))
    resolved = registry.resolve("researcher")
    assert resolved.authorization_header == "Bearer secret-token"
    assert "secret-token" not in repr(resolved)
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_runtime.py -q
```

Expected: FAIL because storage methods and `A2AAgentRegistry` are absent.

- [ ] **Step 3: Add durable tables**

Add these tables in `Storage._init`:

```sql
CREATE TABLE IF NOT EXISTS a2a_tasks (
    team_run_id TEXT NOT NULL,
    logical_task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    business_attempt INTEGER NOT NULL,
    execution_id TEXT NOT NULL UNIQUE,
    idempotency_key TEXT NOT NULL UNIQUE,
    transport_retry_count INTEGER NOT NULL DEFAULT 0,
    context_id TEXT NOT NULL,
    remote_task_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (team_run_id, logical_task_id, business_attempt)
);
CREATE TABLE IF NOT EXISTS a2a_artifacts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
```

Methods must use parameterized SQL and return dictionaries consistent with other Storage APIs.

- [ ] **Step 4: Implement registry validation**

`A2AAgentRegistry.resolve` must reject unknown/disabled roles, missing URL, non-HTTP(S) URLs, URL credentials, missing token environment variables, and Builder configurations with `allow_local_fallback=True`. It must also reject plain HTTP for non-loopback hosts unless the development-only global `allow_insecure_http=True` override was explicitly configured.

Return:

```python
@dataclass(frozen=True, repr=False)
class ResolvedA2AAgent:
    role: str
    base_origin: str
    agent_card_url: str
    authorization_header: str
    allow_local_fallback: bool
    workspace_root: str

    def __repr__(self) -> str:
        return f"ResolvedA2AAgent(role={self.role!r}, origin={self.base_origin!r})"
```

- [ ] **Step 5: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_runtime.py tests\test_storage_session.py -q
```

Expected: selected tests pass and existing Storage behavior remains compatible.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/storage.py src/a2a_runtime.py tests/test_a2a_runtime.py
git commit -m "feat: persist A2A task mappings"
```

---

### Task 5: Implement the A2A SDK client and protected artifact transport

**Files:**

- Modify: `src/a2a_runtime.py`
- Modify: `src/member_execution.py`
- Modify: `tests/test_a2a_runtime.py`
- Modify: `tests/test_member_execution.py`

**Interfaces:**

- Produces `A2AClientRuntime.execute(request: MemberTaskRequest) -> MemberExecutionResult`.
- Produces `A2AArtifactTransport.download(manifest: RemoteArtifactManifest, agent: ResolvedA2AAgent, session_id: str) -> dict[str, Any]`.
- Produces `A2AMemberExecutor(MemberExecutor)`.
- Uses official APIs `A2ACardResolver`, `ClientConfig`, `create_client`, `Message`, `Part`, `Role`, and `SendMessageRequest` only inside `src/a2a_runtime.py`.

- [ ] **Step 1: Write failing client, recovery, and artifact tests**

Cover behaviors, not SDK mock call counts:

```python
def test_remote_executor_reuses_persisted_task_after_timeout(fake_a2a_transport, storage):
    storage.upsert_a2a_task(
        team_run_id="run_1",
        logical_task_id="build",
        role="builder",
        business_attempt=0,
        execution_id="exec_build_1",
        idempotency_key="run_1:build:0",
        transport_retry_count=1,
        context_id="ctx_existing",
        remote_task_id="task_existing",
        endpoint="http://127.0.0.1:8102",
        status="working",
    )
    executor = _remote_executor(fake_a2a_transport, storage)
    result = executor.execute(_member_request(role="builder"))
    assert result.remote_task_id == "task_existing"
    assert fake_a2a_transport.submissions == []


def test_builder_transport_retry_keeps_stable_execution_identity(fake_a2a_transport, storage):
    request = _member_request(role="builder", execution_id="exec_build_1")
    executor = _remote_executor(fake_a2a_transport, storage)
    executor.execute(request)
    executor.execute(request)
    rows = storage.list_a2a_tasks(execution_id="exec_build_1")
    assert len(rows) == 1
    assert rows[0]["transport_retry_count"] == 1


def test_artifact_download_rejects_cross_origin_redirect(tmp_path):
    transport = _artifact_transport(
        tmp_path,
        response=_redirect("https://evil.example/file"),
    )
    with pytest.raises(A2ARuntimeError, match="artifact origin"):
        transport.download(_manifest(), _resolved_agent(), "session_1")


def test_artifact_download_rejects_hash_mismatch(tmp_path):
    transport = _artifact_transport(tmp_path, response=_bytes_response(b"tampered"))
    with pytest.raises(A2ARuntimeError, match="sha256"):
        transport.download(_manifest(sha256="0" * 64), _resolved_agent(), "session_1")
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_runtime.py tests\test_member_execution.py -q
```

Expected: FAIL because client execution and artifact download are absent.

- [ ] **Step 3: Implement the official SDK adapter**

Inside `A2AClientRuntime._execute_async`:

```python
async with httpx.AsyncClient(
    headers={"Authorization": agent.authorization_header},
    timeout=self.request_timeout,
    follow_redirects=False,
) as http_client:
    self.trace_provider.inject(http_client.headers)
    resolver = A2ACardResolver(http_client, agent.base_origin)
    card = await resolver.get_agent_card()
    self._validate_card(card, request.role, agent.base_origin)
    client = await create_client(
        card,
        client_config=ClientConfig(
            httpx_client=http_client,
            supported_protocol_bindings=["HTTP+JSON"],
        ),
    )
    message = Message(
        role=Role.ROLE_USER,
        message_id=str(uuid4()),
        context_id=context_id,
        task_id=remote_task_id or None,
        parts=[Part(text=json.dumps(request_payload, ensure_ascii=False))],
    )
    stream = client.send_message(SendMessageRequest(message=message))
    result = await self._consume_events(stream, request, agent)
    await client.close()
    return result
```

Keep all protobuf/SDK event inspection in `_consume_events`; translate terminal states immediately to project status and reject unknown transitions with `protocol_error`.

`MemberTaskRequest.execution_id` and its derived `idempotency_key` stay stable for one logical business attempt. Persist the mapping before submission. Transport retries only increment `transport_retry_count`, query/resume the same A2A task, and never create a new Builder submission. A new business `attempt` is the only event that may create a new execution ID.

Inject the active W3C `traceparent` and optional `tracestate` into the authenticated `httpx` client before Agent Card/task/artifact calls. Do not serialize these headers into task payloads or persistence.

- [ ] **Step 4: Implement artifact validation**

Define:

```python
@dataclass(frozen=True)
class RemoteArtifactManifest:
    artifact_id: str
    name: str
    url: str
    mime_type: str
    size_bytes: int
    sha256: str
```

Allow only `text/plain`, `text/markdown`, `application/json`, `text/x-diff`, `application/zip`, and `application/octet-stream`. Stream to a temporary file inside `outputs/a2a/artifacts`, stop before exceeding `artifact_max_bytes`, verify SHA-256, then atomically rename and register the artifact in Storage.

- [ ] **Step 5: Add A2AMemberExecutor**

```python
class A2AMemberExecutor:
    def __init__(self, client: A2AClientRuntime) -> None:
        self.client = client

    def execute(self, request: MemberTaskRequest) -> MemberExecutionResult:
        return self.client.execute(request)
```

Do not put role fallback inside this class; Lead/Team owns fallback policy.

- [ ] **Step 6: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_runtime.py tests\test_member_execution.py -q
```

Expected: selected tests pass, including timeout recovery and hostile artifact cases.

- [ ] **Step 7: Commit Task 5**

```powershell
git add src/a2a_runtime.py src/member_execution.py tests/test_a2a_runtime.py tests/test_member_execution.py
git commit -m "feat: add A2A client and artifact validation"
```

---

### Task 6: Serve each role as an authenticated A2A HTTP process

**Files:**

- Create: `src/a2a_service.py`
- Create: `tests/test_a2a_service.py`
- Modify: `src/main.py:38-138`

**Interfaces:**

- Produces `WorkflowAgentA2AExecutor(AgentExecutor)`.
- Produces `build_role_agent_card(role: str, base_url: str) -> AgentCard`.
- Produces `build_role_app(config: AppConfig, role: str, base_url: str, runtime_factory=None) -> FastAPI`.
- Produces `serve_role(config, role, host, port) -> None`.

- [ ] **Step 1: Write failing Agent Card, auth, and role execution tests**

```python
def test_researcher_agent_card_advertises_only_research_capabilities():
    card = build_role_agent_card("researcher", "http://127.0.0.1:8101")
    assert [skill.id for skill in card.skills] == ["workflow-agent-researcher"]
    assert card.supported_interfaces[0].protocol_binding == "HTTP+JSON"


def test_a2a_route_requires_bearer_token(test_client):
    response = test_client.post("/a2a/rest/v1/message:send", json={})
    assert response.status_code == 401


def test_agent_card_is_public(test_client):
    response = test_client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    assert "token" not in response.text.lower()


def test_remote_span_extracts_w3c_parent(test_client, trace_recorder):
    response = test_client.post(
        "/a2a/rest/v1/message:send",
        headers={"Authorization": "Bearer test", "traceparent": _traceparent("abc123")},
        json=_valid_send_request(),
    )
    assert response.status_code == 200
    assert trace_recorder.remote_trace_id == "abc123"
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_service.py -q
```

Expected: FAIL because the service module is absent.

- [ ] **Step 3: Build role Agent Cards and routes using official SDK APIs**

Use `AgentCard`, `AgentInterface`, `AgentProvider`, `AgentCapabilities`, and one `AgentSkill` per role. Expose only HTTP+JSON at `/a2a/rest`; build routes with `create_agent_card_routes`, `create_rest_routes`, and `add_a2a_routes_to_fastapi`.

Create a FastAPI middleware that leaves only the Agent Card and health endpoint public. Compare the supplied token with `secrets.compare_digest` and never include it in error bodies. The middleware extracts W3C `traceparent`/`tracestate` into the OpenTelemetry context before creating the remote `a2a.task`/`member.run` spans and rejects malformed headers without echoing them.

- [ ] **Step 4: Implement WorkflowAgentA2AExecutor**

`execute` must parse the JSON text from `context.get_user_input()`, validate it into `MemberTaskRequest`, enforce that the requested role matches the service role, replace any caller-supplied workspace value with the service's configured absolute `workspace_root`, and call `AgentRuntime` through `asyncio.to_thread`.

Publish progress and completion with `TaskUpdater`:

```python
updater = TaskUpdater(
    event_queue=event_queue,
    task_id=context.task_id or "",
    context_id=context.context_id or "",
)
await updater.start_work(
    message=updater.new_agent_message(parts=[Part(text="working")])
)
result = await asyncio.to_thread(self.local_executor.execute, request)
await updater.add_artifact(
    parts=[Part(text=json.dumps(self._result_payload(result), ensure_ascii=False))],
    name="member-result",
    last_chunk=True,
)
await updater.complete()
```

`cancel` must publish a canceled terminal state. It must not claim filesystem rollback.

- [ ] **Step 5: Add the protected artifact endpoint**

Expose `GET /artifacts/{artifact_id}`. Resolve only entries registered in `a2a_artifacts`, verify the resolved file remains inside the configured output directory, and return `FileResponse` with the recorded MIME and SHA-256 response header.

- [ ] **Step 6: Add service CLI parsing only**

Add `--serve-a2a-role {researcher,builder,reviewer}`, `--host`, and `--port` arguments to `main.py`. Dispatch to `serve_role`; do not yet add remote Lead CLI commands.

- [ ] **Step 7: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_service.py tests\test_cli_json_args.py -q
```

Expected: service and existing CLI argument tests pass.

- [ ] **Step 8: Commit Task 6**

```powershell
git add src/a2a_service.py src/main.py tests/test_a2a_service.py
git commit -m "feat: serve workflow roles over A2A"
```

---

### Task 7: Route Team DAG members remotely with role-specific fallback

**Files:**

- Modify: `src/team.py:42-130`
- Modify: `src/team.py:313-566`
- Create: `tests/test_team_a2a.py`

**Interfaces:**

- Produces `TeamRuntime._executor_for(request: MemberTaskRequest) -> MemberExecutor`.
- Consumes `LocalMemberExecutor`, `A2AMemberExecutor`, `A2AAgentRegistry`, and `TraceProvider`.
- Preserves existing task, Evidence Gate, message, event, and synthesis payload formats.

- [ ] **Step 1: Write failing remote-routing and fallback tests**

```python
def test_team_routes_enabled_researcher_to_remote_executor(team_runtime):
    result = team_runtime.run(
        "分析项目架构",
        "default-dev-team",
        auto_approve=True,
        no_memory=True,
    )
    researcher = next(item for item in result.member_runs if item["member_name"] == "researcher")
    assert researcher["result"]["execution_mode"] == "a2a"


def test_researcher_failure_falls_back_locally(team_runtime_with_failed_researcher):
    result = team_runtime_with_failed_researcher.run(
        "分析项目架构", "default-dev-team", auto_approve=True, no_memory=True
    )
    researcher = next(item for item in result.member_runs if item["member_name"] == "researcher")
    assert researcher["result"]["execution_mode"] == "local_fallback"


def test_builder_failure_never_falls_back_locally(team_runtime_with_failed_builder):
    result = team_runtime_with_failed_builder.run(
        "修改 README", "default-dev-team", auto_approve=True, no_memory=True
    )
    builder = next(item for item in result.member_runs if item["member_name"] == "builder")
    assert builder["status"] in {"failed", "blocked"}
    assert builder["result"]["execution_mode"] == "a2a"


def test_lead_restart_resumes_same_builder_execution_id(restarted_team_runtime):
    result = restarted_team_runtime.resume("run_1")
    builder = next(item for item in result.member_runs if item["member_name"] == "builder")
    assert builder["result"]["execution_id"] == "exec_build_1"
    assert restarted_team_runtime.remote_transport.builder_submissions == 1
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_team_a2a.py -q
```

Expected: FAIL because TeamRuntime does not route through A2A.

- [ ] **Step 3: Route per member role**

Create `MemberTaskRequest` from the stored runtime task item and member session. Derive and persist a stable `execution_id`/`idempotency_key` from `team_run_id + logical_task_id + business_attempt`; never include `transport_retry_count` in either identity. `_executor_for` returns A2A only when global A2A and the role entry are both enabled; otherwise it returns Local.

Wrap each member execution in `a2a.task` or `member.run` spans with only safe IDs, role, attempt, and status attributes.

- [ ] **Step 4: Implement bounded retry and fallback at the Lead**

For retryable A2A errors, reuse the persisted task mapping and increment only `transport_retry_count`. Query/resume the same remote task after timeout and across Lead restart; do not submit another task when a mapping exists. After `max_retries`:

```python
if request.role in {"researcher", "reviewer"} and remote.allow_local_fallback:
    fallback = self.local_member_executor.execute(request)
    fallback.execution_mode = "local_fallback"
    return fallback
return MemberExecutionResult(
    member_name=request.member_name,
    role=request.role,
    status="blocked" if request.role == "builder" else "failed",
    final_answer="",
    acceptance={"passed": False, "issues": [error.code]},
    tool_calls=[],
    artifacts=[],
    events=[],
    execution_mode="a2a",
    error=str(error),
)
```

Do not let fallback bypass the existing Evidence Gate.

- [ ] **Step 5: Verify GREEN and local compatibility**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_team_a2a.py tests\test_team_mode.py -q
```

Expected: all remote policy and existing local Team tests pass.

- [ ] **Step 6: Commit Task 7**

```powershell
git add src/team.py tests/test_team_a2a.py
git commit -m "feat: route team roles through A2A"
```

---

### Task 8: Add governed LangMem candidate extraction

**Files:**

- Create: `src/langmem_runtime.py`
- Create: `tests/test_langmem_runtime.py`
- Modify: `src/memory_runtime.py:452-713`
- Modify: `src/team.py:803-854`

**Interfaces:**

- Produces `LangMemCandidateExtractor.extract(...) -> list[MemoryCandidate]`.
- Uses LangMem `create_memory_manager`, never `create_memory_store_manager`.
- Consumes `MemoryGovernanceRuntime.policy.evaluate` as the only persistence/activation path.

- [ ] **Step 1: Write failing extraction and governance tests**

```python
def test_langmem_converts_structured_output_to_role_namespace(fake_langmem_manager):
    extractor = LangMemCandidateExtractor(
        manager_factory=lambda *_args, **_kwargs: fake_langmem_manager,
        max_candidates=8,
    )
    candidates = extractor.extract(
        role="researcher",
        team_run_id="run_1",
        task_id="task_1",
        final_answer="The runtime uses LangGraph.",
        accepted_evidence={"ev_1": _evidence_card("ev_1")},
        accepted_source_refs={"a2a:researcher:task_1"},
    )
    assert candidates[0].namespace == "project:team:researcher"
    assert candidates[0].evidence_ids == ["ev_1"]


def test_unverified_remote_inference_cannot_become_active(tmp_path):
    runtime = MemoryGovernanceRuntime(tmp_path / "memory.sqlite")
    candidate = MemoryCandidate(
        memory_type="semantic",
        namespace="project:team:researcher",
        subject="remote inference",
        content="unsupported claim",
        trust="model_inference",
        confidence=0.4,
        evidence_ids=[],
        source_refs=["a2a:researcher:task_1"],
    )
    record = runtime.evaluate_candidates([candidate])[0]
    assert record.status == "candidate"


def test_each_candidate_keeps_only_its_validated_evidence(fake_langmem_manager):
    candidates = _extract(
        fake_langmem_manager.results(
            _memory("supported", evidence_ids=["ev_1"]),
            _memory("unsupported", evidence_ids=["missing"]),
        ),
        accepted_evidence={"ev_1": _evidence_card("ev_1")},
    )
    assert candidates[0].evidence_ids == ["ev_1"]
    assert candidates[0].trust == "evidence"
    assert candidates[1].evidence_ids == []
    assert candidates[1].trust == "model_inference"
    record = MemoryGovernanceRuntime().evaluate_candidates([candidates[1]])[0]
    assert record.status == "candidate"
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langmem_runtime.py -q
```

Expected: FAIL because the extractor and batch evaluation entry point do not exist.

- [ ] **Step 3: Define a structured LangMem schema and extractor**

```python
class ExtractedAgentMemory(BaseModel):
    memory_type: Literal["semantic", "episodic", "procedural", "preference"]
    subject: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class LangMemCandidateExtractor:
    def extract(
        self,
        *,
        role: str,
        team_run_id: str,
        task_id: str,
        final_answer: str,
        accepted_evidence: dict[str, EvidenceCard],
        accepted_source_refs: set[str],
    ) -> list[MemoryCandidate]:
        manager = self.manager_factory(
            self.model,
            schemas=[ExtractedAgentMemory],
            enable_inserts=True,
            enable_updates=True,
            enable_deletes=False,
        )
        extracted = manager.invoke(
            {"messages": [{"role": "assistant", "content": final_answer}]}
        )
        return self._to_candidates(
            extracted[: self.max_candidates], role, team_run_id, task_id,
            accepted_evidence, accepted_source_refs
        )
```

Each extracted item must cite its own evidence IDs and source refs. Intersect those claims with the Lead-accepted Evidence Cards/source refs; never copy the full accepted list onto every candidate. Clamp confidence to at most `0.4` when a candidate has no validated evidence. Use trust `evidence` only when that candidate has validated evidence and `model_inference` otherwise; unmatched claims remain `candidate` under the existing policy.

- [ ] **Step 4: Add governed batch evaluation**

Add this method to `MemoryGovernanceRuntime`:

```python
def evaluate_candidates(
    self, candidates: list[MemoryCandidate], session_id: str = ""
) -> list[MemoryRecord]:
    records = [self.policy.evaluate(candidate) for candidate in candidates]
    if self.storage and session_id:
        for record in records:
            event = "memory.promoted" if record.status == "active" else "memory.candidate.created"
            self.storage.add_event(session_id, event, asdict(record))
    return records
```

- [ ] **Step 5: Wire extraction after Team synthesis**

After Evidence Gate and Lead synthesis, call LangMem only when `config.langmem.enabled`. Supply the accepted Evidence Card map and accepted source-ref set so attribution can be checked per candidate. Catch extraction errors, emit `memory_extraction_error`, and continue through the existing `consolidate_run` path when `fallback_to_rule_consolidator` is true.

- [ ] **Step 6: Verify GREEN and memory regressions**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langmem_runtime.py tests\test_memory_runtime.py tests\test_team_mode.py -q
```

Expected: all selected tests pass and legacy consolidation remains active when LangMem is disabled.

- [ ] **Step 7: Commit Task 8**

```powershell
git add src/langmem_runtime.py src/memory_runtime.py src/team.py tests/test_langmem_runtime.py
git commit -m "feat: extract governed memories with LangMem"
```

---

### Task 9: Instrument Tool, MCP, Context, and Team spans

**Files:**

- Modify: `src/tool_runtime.py:31-304`
- Modify: `src/mcp_runtime.py:37-240`
- Modify: `src/context_harness.py:273-531`
- Modify: `src/team.py`
- Modify: `tests/test_tool_runtime.py`
- Modify: `tests/test_mcp_runtime.py`
- Modify: `tests/test_context_harness.py`
- Modify: `tests/test_observability.py`

**Interfaces:**

- Consumes shared `TraceProvider`; no module initializes its own exporter.
- Adds safe spans `tool.run`, `mcp.call`, `context.prepare`, `memory.retrieve`, `team.plan`, and `answer.synthesize`.

- [ ] **Step 1: Write failing safe-attribute tests**

```python
def test_tool_span_records_schema_hash_without_arguments(recording_trace_provider):
    runtime = _tool_runtime(trace_provider=recording_trace_provider)
    runtime.run(_tool_request("read_file", {"path": "README.md"}))
    span = recording_trace_provider.one("tool.run")
    assert span.attributes["tool.name"] == "read_file"
    assert "tool.schema_hash" in span.attributes
    assert "path" not in span.attributes


def test_mcp_span_does_not_record_tool_arguments(recording_trace_provider):
    manager = _mcp_manager(trace_provider=recording_trace_provider)
    manager.call_tool("fetch", "fetch", {"url": "https://secret.example"}, _context())
    span = recording_trace_provider.one("mcp.call")
    assert span.attributes["mcp.server"] == "fetch"
    assert "url" not in span.attributes
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tool_runtime.py tests\test_mcp_runtime.py tests\test_context_harness.py tests\test_observability.py -q
```

Expected: new tracing assertions fail because these modules do not accept a provider.

- [ ] **Step 3: Inject one shared provider**

Add optional `trace_provider` constructor arguments defaulting to `NoOpTraceProvider`. Wrap operations with safe attributes only. Record result status, duration, provider name, schema hash, artifact hash, counts, and IDs; never record raw args, tool output, prompt text, or memory content.

- [ ] **Step 4: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_tool_runtime.py tests\test_mcp_runtime.py tests\test_context_harness.py tests\test_observability.py -q
```

Expected: selected tests pass with no secrets or raw tool arguments in spans.

- [ ] **Step 5: Commit Task 9**

```powershell
git add src/tool_runtime.py src/mcp_runtime.py src/context_harness.py src/team.py tests/test_tool_runtime.py tests/test_mcp_runtime.py tests/test_context_harness.py tests/test_observability.py
git commit -m "feat: trace agent tools and context"
```

---

### Task 10: Add operator CLI, real-process tests, and deployment documentation

**Files:**

- Modify: `src/main.py`
- Modify: `pytest.ini`
- Create: `tests/test_a2a_real.py`
- Modify: `README.md`

**Interfaces:**

- Produces CLI actions `--a2a-health`, `--a2a-agents`, `--a2a-run-role`, and `--doctor-observability`.
- Keeps `--serve-a2a-role`, `--host`, and `--port` from Task 6.
- Uses `RUN_REAL_A2A=1` and `RUN_REAL_OTLP=1` as explicit real-integration gates.

- [ ] **Step 1: Write failing CLI health tests**

```python
def test_a2a_health_reports_each_configured_role(monkeypatch, capsys):
    monkeypatch.setattr("src.main.AgentRuntime", _runtime_with_a2a_health())
    exit_code = main_with_args(["--a2a-health", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert {item["role"] for item in payload["items"]} == {
        "researcher",
        "builder",
        "reviewer",
    }


def test_observability_doctor_reports_jsonl_fallback(capsys):
    exit_code = main_with_args(["--doctor-observability", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["jsonl_fallback"]
```

- [ ] **Step 2: Run and verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_main_config.py tests\test_a2a_real.py -q
```

Expected: CLI behavior tests fail because commands are not wired.

- [ ] **Step 3: Add health, inventory, direct-role, and trace doctor commands**

`--a2a-health` resolves Agent Cards without submitting tasks. `--a2a-agents` prints sanitized configured inventory. `--a2a-run-role ROLE` executes one role through `MemberExecutor` for diagnosis. `--doctor-observability` reports provider type, JSONL path, and whether an OTLP endpoint is configured, never the endpoint credentials or headers.

- [ ] **Step 4: Add opt-in real integration markers**

Extend `pytest.ini`:

```ini
    a2a: real A2A HTTP process tests
    observability: real OTLP collector tests
```

`tests/test_a2a_real.py` must skip unless `RUN_REAL_A2A=1`. When enabled, launch three hidden subprocesses on dynamically reserved localhost ports with three distinct temporary workspace roots, wait for health with a bounded timeout, run one Team task, and terminate only those exact child processes in `finally`. Assert a Lead-created trace ID is preserved in the remote Agent span, the three workspaces remain isolated, and restarting Lead resumes the same Builder execution ID without a duplicate submission.

- [ ] **Step 5: Document deployment and security**

Add README sections covering:

- dependency installation and the MCP `<2` compatibility note;
- three service commands with separate absolute workspace roots;
- Bearer token environment variables;
- loopback-only HTTP and HTTPS-by-default for non-loopback endpoints; document `allow_insecure_http` as development-only;
- sanitized `agent_config.json` role entries;
- Lead Team invocation;
- OTLP configuration and JSONL fallback;
- LangMem governance behavior;
- remote failure and Builder non-fallback semantics;
- artifact size/hash/origin restrictions;
- real integration test commands.

- [ ] **Step 6: Run focused and full verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_a2a_runtime.py tests\test_a2a_service.py tests\test_team_a2a.py tests\test_langmem_runtime.py tests\test_observability.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.main --a2a-agents --format json
.\.venv\Scripts\python.exe -m src.main --doctor-observability --format json
```

Expected:

- focused tests pass;
- full suite exits 0 with only explicitly gated real integrations skipped;
- both doctor commands emit valid JSON without credentials.

- [ ] **Step 7: Inspect the final diff for secrets and unrelated files**

```powershell
git diff --check
git diff --name-only
rg -n "Bearer [A-Za-z0-9]|sk-[A-Za-z0-9]|A2A_.*TOKEN=. +" src tests README.md agent_config.json.example .env.example
```

Expected: `git diff --check` is clean, changed files match this plan, and the secret scan returns no credential values.

- [ ] **Step 8: Commit Task 10**

```powershell
git add src/main.py pytest.ini tests/test_a2a_real.py README.md
git commit -m "docs: add A2A deployment workflow"
```

---

## Final Acceptance Checklist

- [ ] Three independent role services expose valid Agent Cards and authenticated HTTP+JSON A2A endpoints.
- [ ] Each role uses its configured independent absolute workspace root; reads and writes cannot escape it.
- [ ] Lead discovers, authenticates, submits, resumes, and records remote tasks.
- [ ] One business attempt retains a stable execution ID across transport retry and Lead restart; Builder is not submitted twice.
- [ ] Researcher and Reviewer degrade locally only after bounded retries.
- [ ] Builder never silently falls back to the Lead working directory.
- [ ] Remote artifacts pass origin, redirect, size, MIME, and SHA-256 validation before Evidence ingestion.
- [ ] Remote output cannot modify Lead DAG, permissions, system instructions, or other role configuration.
- [ ] LangMem creates candidates only; all persistence goes through `MemoryPolicyGate`.
- [ ] OpenInference/OpenTelemetry spans cover Runtime, Team, A2A, Tool, MCP, Context, Memory, and Synthesis.
- [ ] W3C `traceparent`/`tracestate` preserves the Trace ID across Lead and role processes.
- [ ] Trace and event payloads contain no tokens, API keys, raw sensitive tool args, or complete memory content.
- [ ] OTLP and LangMem failures follow their approved non-critical fallback paths.
- [ ] Existing local/offline Team, MCP v1, Skill, Eval, and Benchmark behavior remains compatible.
- [ ] Full pytest suite passes with only explicitly gated real integration tests skipped.
