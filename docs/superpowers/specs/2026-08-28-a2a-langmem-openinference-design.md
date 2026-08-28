# A2A、LangMem 与 OpenInference 集成设计

## 1. 目标

在保留现有 `AgentRuntime`、`TeamRuntime`、任务 DAG、权限门禁、Evidence Gate 和 Memory Governance 的前提下，增加三项能力：

1. 将 Researcher、Builder、Reviewer 拆分为可跨机器部署的独立 A2A HTTP Agent。
2. 使用 LangMem 从运行结果中提取和整合候选记忆，但继续由现有 `MemoryPolicyGate` 决定记忆是否激活、替换或失效。
3. 使用 OpenInference 和 OpenTelemetry 为 LangGraph、LLM、Tool、MCP、A2A、Memory 与 Synthesis 建立统一 Trace，并在 OTLP 不可用时降级到本地 JSONL。

本设计采用 Adapter-first 方式，不用其他 Agent 框架替换 LangGraph，也不允许远程 Agent 绕过 Lead、权限或证据控制面。

## 2. 已确认约束

- Researcher、Builder、Reviewer 以独立 HTTP 进程部署，可位于不同机器；非回环地址默认必须使用 HTTPS。
- Lead/`TeamRuntime` 是 A2A 客户端和唯一 DAG 编排者。
- A2A 服务使用静态 Bearer Token；配置只保存 Token 环境变量名。
- 每个角色服务使用独立的绝对工作区根目录，不共享网络目录，也不依赖进程 CWD。
- 小结果内联；大文件通过受 Bearer Token 保护的 Artifact URL 返回。
- Researcher/Reviewer 远程失败后允许本地降级；Builder 不允许自动本地降级。
- 长期记忆由 Lead 集中治理，并按角色隔离命名空间。
- Trace 优先导出到标准 OTLP Endpoint，未配置或不可用时写入本地 JSONL。
- Phoenix 不是 Runtime 核心依赖。
- 第一阶段不引入 Mem0、Graphiti、Neo4j、外部向量数据库或远程代码执行服务。

## 3. 复用决策

| 能力                      | 候选              | 决策                                            |
| ------------------------- | ----------------- | ----------------------------------------------- |
| Agent 互操作              | 官方 `a2a-sdk`    | 封装采用，隐藏在项目自有 A2A Adapter 后         |
| 记忆提取与整合            | `langmem`         | 封装采用，只输出 `MemoryCandidate`              |
| 标准 Trace                | OpenTelemetry SDK | 直接采用，通过项目自有 `TraceProvider` 使用     |
| Agent/LLM Instrumentation | OpenInference     | 封装采用，业务模块不直接依赖具体 exporter       |
| Trace UI                  | Phoenix           | 可选部署，不加入核心依赖                        |
| 图记忆                    | Graphiti          | 后续单独评估，本阶段不实现                      |
| 沙箱                      | SWE-ReX/E2B       | 本阶段不实现，继续使用现有 Docker Skill Runtime |

项目根目录当前没有许可证文件。在分发新依赖或复制参考实现前，需要单独确定项目许可证。本设计只允许调用 MIT 或 Apache-2.0 兼容 SDK 的公开 API，不复制许可证不明确的源码。

## 4. 总体架构

```text
TeamRuntime / Lead
  |
  +-- AssignmentPlanner -> TaskGraph
  |
  +-- MemberExecutorRouter
  |     +-- LocalMemberExecutor -> AgentRuntime
  |     +-- A2AMemberExecutor
  |           +-- Agent Card discovery
  |           +-- Bearer authentication
  |           +-- retry / timeout / resume
  |           +-- Artifact download and validation
  |
  +-- Evidence Gate
  +-- LangMemCandidateExtractor
  +-- MemoryPolicyGate
  +-- LeadSynthesizer
```

引入 `MemberExecutor` 接口隔离本地和远程成员执行。`TeamRuntime` 只依赖此接口，不直接依赖 A2A SDK。

### 4.1 MemberExecutor

`MemberExecutor` 接收逻辑任务、成员配置、绝对工作区根目录、运行上下文和超时配置，返回统一的 `MemberExecutionResult`。工作区根目录沿 `MemberTaskRequest -> RuntimeOptions -> Agent state -> ToolContext` 显式传播；所有文件工具必须做根目录包含检查并拒绝路径穿越和符号链接逃逸。结果必须包含：

- 执行状态；
- 最终回答或错误；
- Tool Call 摘要；
- Evidence Card；
- Artifact 描述；
- 远程任务标识；
- Trace 关联信息；
- 候选 Memory Source 信息。

`LocalMemberExecutor` 包装现有 `AgentRuntime`。`A2AMemberExecutor` 负责协议转换、认证、重试、状态恢复和 Artifact 获取。

### 4.2 角色服务

Researcher 服务：

- 使用独立工作目录；
- 允许 read、list、search、RAG 和经过权限批准的 MCP；
- 禁止写项目文件；
- 返回来源、查询、文件证据和研究结论。

Builder 服务：

- 使用独立工作目录；
- 读写和测试操作继续经过其本地 `ToolRuntime` 与权限规则；
- 返回修改文件清单、补丁或变更 Artifact、测试证据和未解决阻塞；
- 失败后不在 Lead 工作区自动重做。

Reviewer 服务：

- 使用独立工作目录或 Lead 提供的只读 Artifact；
- 校验需求、Evidence、变更和测试结果；
- 不修改 Builder 产物；
- 返回验收结论、风险和缺失证据。

## 5. A2A 协议设计

### 5.1 发现与注册

`A2AAgentRegistry` 从配置读取角色端点、Agent Card URL、Token 环境变量名、是否启用和是否允许本地降级。Agent Card 被缓存，并在 TTL 到期、协议错误或能力变化时刷新。

Lead 必须验证：

- Agent Card 的角色与配置角色一致；
- 必需 Skill/Capability 存在；
- 协议版本受支持；
- Agent Card Origin 与配置 Origin 一致；
- Builder 明确声明可产生变更 Artifact。

### 5.2 任务转换

每个 `TaskNode` 转成 A2A Message，包含：

- 用户目标；
- 角色系统约束；
- Required Tools；
- Required Evidence；
- Acceptance Criteria；
- 输入 Artifact 引用；
- Deadline；
- `team_run_id`、逻辑任务 ID、业务 Attempt、稳定 `execution_id` 和幂等键。

远程 Agent 无权修改 Lead 的 DAG、权限规则、系统约束或其他成员配置。

### 5.3 标识与幂等

关联链为：

```text
team_run_id
  -> task_id
    -> a2a_context_id
      -> a2a_task_id
        -> trace_id / span_id
```

Lead 使用 `team_run_id + logical_task_id + business_attempt` 生成稳定的 `execution_id` 和幂等键。业务 Attempt 与传输重试严格分离：网络重试只增加 `transport_retry_count`，不得改变执行身份。Lead 在首次发送前持久化映射；网络超时或 Lead 重启后必须按同一 `execution_id` 查询并恢复已有远程任务，不能直接重复提交可能产生副作用的 Builder 任务。只有新的业务 Attempt 才能生成新的执行身份。

### 5.4 状态映射

A2A 状态统一映射到项目状态：

| A2A 状态            | 项目状态                      |
| ------------------- | ----------------------------- |
| submitted / working | running                       |
| input-required      | blocked，记录所需输入         |
| completed           | completed，进入 Evidence Gate |
| failed              | failed                        |
| canceled            | canceled                      |
| rejected            | denied                        |

非法状态转换产生 `protocol_error`，不得被解释为成功。

## 6. 认证与安全

- Bearer Token 只通过环境变量读取。
- 明文 HTTP 仅允许回环地址。非回环端点必须使用 HTTPS；开发环境如确需明文连接，必须显式设置默认关闭的 `allow_insecure_http`，并产生安全告警。
- Agent Card、配置导出、事件、Trace、错误和 Memory 中不得出现 Token。
- 所有异常在持久化前经过统一 Secret Redactor。
- A2A 响应、Agent Card、远程文本和 Artifact 一律视为不可信输入。
- 第一阶段不允许远程 Agent 再委派其他 Agent。
- Lead 不接受远程权限规则、系统提示或 DAG 更新。
- Builder 的写入仍由 Builder 本地 `PermissionRule` 和 `ToolRuntime` 控制。
- A2A HTTP 服务应支持只监听指定 Host；默认示例监听 `127.0.0.1`，跨机器部署由用户显式配置 TLS 与网络边界。

## 7. Artifact 传输

小型文本、结构化 Evidence 和状态以内联 Part 返回。大型 Artifact 使用带以下元数据的 URL：

- 文件名；
- MIME；
- 字节数；
- SHA-256；
- 下载 URL；
- 过期时间；
- Evidence ID 或 Source Ref。

`ArtifactTransport` 必须执行：

1. URL Origin 与角色配置 Origin 一致；
2. 禁止任意跨域重定向；
3. 限制连接、读取和总任务超时；
4. 默认最大文件为 50 MiB；
5. MIME 和扩展名满足白名单；
6. 流式计算并验证 SHA-256；
7. 下载到项目输出目录内的临时文件，再注册为 Context Artifact；
8. 校验失败的文件不得进入 Evidence 或 Memory。

## 8. 重试、恢复与降级

- 默认网络重试次数为 2。
- 重试仅覆盖可恢复网络错误、部分 5xx 和暂时不可用状态。
- 认证、协议、能力不匹配和 Artifact 校验错误不自动重试。
- Researcher/Reviewer 重试耗尽后可使用 `LocalMemberExecutor` 降级。
- Builder 重试耗尽后标记当前任务失败或阻塞，依赖任务按现有 DAG 规则处理。
- Lead 持久化 A2A Context/Task ID，使进程恢复后能够继续轮询原远程任务。
- OTLP 导出和 LangMem 提取失败不能使 Agent 主任务失败。

## 9. LangMem 集成

### 9.1 所有权

现有 `MemoryGovernanceRuntime` 是唯一长期记忆写入口。LangMem 不直接访问或修改 `memory.sqlite`。

```text
MemberReport / Evidence / ToolResult
                 |
                 v
       LangMemCandidateExtractor
                 |
                 v
 semantic / episodic / procedural / preference candidates
                 |
                 v
          MemoryPolicyGate
                 |
                 v
 active / candidate / superseded / invalidated
```

### 9.2 命名空间

- `project:team:researcher`
- `project:team:builder`
- `project:team:reviewer`

Lead 可以在 Context Assembly 时跨角色检索，但不复制原始记录，也不允许角色服务直接读取其他角色的全部长期记忆。

### 9.3 激活规则

- 无 Evidence 的远程推断最多成为 candidate。
- LangMem 每个候选必须声明自己的 Evidence ID/Source Ref；Lead 只保留与已验收 Evidence Card 相交的引用，禁止把本轮全部 Evidence 批量赋给每个候选。
- 候选声明的 Evidence 无法匹配时，清空其 Evidence 归因、信任级别降为 `model_inference`、置信度上限为 0.4，并保持 candidate。
- 工具结果生成的 semantic candidate 必须关联 Tool Evidence。
- Reviewer 验收通过且证据完整的经验才可成为 active procedural memory。
- 明确用户偏好保持最高信任级别。
- 冲突、替换和失效继续由现有 `MemoryPolicyGate` 处理。
- LangMem 不可用或提取失败时，回退现有规则式 `MemoryConsolidator`。

## 10. OpenInference 与 OpenTelemetry

业务代码只依赖项目自有 `TraceProvider`：

```text
TraceProvider
  +-- NoOpTraceProvider
  +-- JsonlTraceProvider
  +-- OpenTelemetryTraceProvider -> OTLP Endpoint
```

Span 层级：

```text
agent.run
  +-- context.prepare
  +-- memory.retrieve
  +-- team.plan
  +-- a2a.task
  |     +-- a2a.discover
  |     +-- a2a.send
  |     +-- a2a.poll
  |     +-- artifact.download
  +-- member.run
  |     +-- llm
  |     +-- tool
  |     +-- mcp
  +-- memory.consolidate
  +-- answer.synthesize
```

允许记录：模型、token、耗时、状态、重试、Tool/MCP Provider、Schema Hash、A2A Agent ID、角色、Evidence ID、Artifact Hash 和验收结果。

禁止记录：API Key、Bearer Token、未脱敏完整 Prompt、敏感工具参数、大文件正文和完整 Memory 内容。

Lead 在所有 A2A HTTP 请求上注入标准 W3C `traceparent` 和可选 `tracestate`；角色服务在创建远程 Span 前提取上下文，使 Lead、远程 Agent、Tool 与 Artifact 共享同一个 Trace ID。传播头不得进入任务正文、持久化记录或日志。

未配置 OTLP Endpoint 时写入 `outputs/traces/agent.jsonl`。OTLP 初始化或导出失败时记录 `trace_export_error` 并继续业务流程。

## 11. 配置设计

```json
{
  "a2a": {
    "enabled": true,
    "request_timeout_seconds": 30,
    "task_timeout_seconds": 600,
    "max_retries": 2,
    "allow_insecure_http": false,
    "artifact_max_bytes": 52428800,
    "agents": {
      "researcher": {
        "enabled": true,
        "agent_card_url": "http://127.0.0.1:8101/.well-known/agent-card.json",
        "token_env": "A2A_RESEARCHER_TOKEN",
        "allow_local_fallback": true
      },
      "builder": {
        "enabled": true,
        "agent_card_url": "http://127.0.0.1:8102/.well-known/agent-card.json",
        "token_env": "A2A_BUILDER_TOKEN",
        "allow_local_fallback": false
      },
      "reviewer": {
        "enabled": true,
        "agent_card_url": "http://127.0.0.1:8103/.well-known/agent-card.json",
        "token_env": "A2A_REVIEWER_TOKEN",
        "allow_local_fallback": true
      }
    }
  },
  "observability": {
    "enabled": true,
    "jsonl_fallback": "outputs/traces/agent.jsonl",
    "capture_prompt_content": false
  },
  "langmem": {
    "enabled": true,
    "fallback_to_rule_consolidator": true
  }
}
```

CLI 增加：

- `--a2a-health`
- `--a2a-agents`
- `--a2a-run-role ROLE`
- `--serve-a2a-role ROLE`
- `--host`
- `--port`
- `--doctor-observability`

## 12. 错误模型

| 错误码                  | 含义                           | 默认处理                 |
| ----------------------- | ------------------------------ | ------------------------ |
| configuration_error     | URL、角色或 Token 环境变量错误 | 启动/调用失败            |
| discovery_error         | Agent Card 不可用或能力不匹配  | 有条件降级               |
| authentication_error    | Bearer Token 被拒绝            | 不重试，不降级 Builder   |
| protocol_error          | 响应或状态转换非法             | 失败并审计               |
| remote_task_error       | 远程角色执行失败               | 按角色降级策略处理       |
| artifact_error          | 来源、大小、类型或哈希非法     | 拒绝 Artifact            |
| timeout                 | 网络或任务超时                 | 查询状态后有限重试       |
| trace_export_error      | OTLP 导出失败                  | JSONL 降级，业务继续     |
| memory_extraction_error | LangMem 提取失败               | 规则式 Consolidator 降级 |

## 13. 测试策略

所有实现遵循 Red-Green-Refactor。

### 13.1 单元测试

- A2A 配置验证和 Secret Redaction；
- Agent Card 角色、Origin、协议和能力检查；
- A2A 状态映射和非法转换；
- 幂等键和恢复映射；
- 业务 Attempt 与传输重试计数分离，Builder 跨超时/Lead 重启不重复提交；
- Artifact Origin、重定向、大小、MIME 和哈希校验；
- Researcher/Reviewer 降级及 Builder 阻塞；
- LangMem 输出到 `MemoryCandidate` 的转换、逐候选 Evidence 归因和治理；
- Trace Span 层级、敏感字段过滤和 JSONL 降级。

### 13.2 契约测试

使用进程内或临时端口的 A2A 测试服务验证官方 SDK 的 Agent Card、认证、任务状态和 Artifact 协议，禁止只验证 Mock 调用次数。

### 13.3 Runtime 集成测试

- 三个角色服务运行于独立测试端口；
- Team DAG 远程完成；
- Researcher/Reviewer 服务不可用时本地降级；
- Builder 服务不可用时任务阻塞；
- Lead 恢复后继续原 A2A Task；
- Trace ID 跨 Lead、远程 Agent、Tool 和 Artifact 保持关联；
- 三个角色的绝对工作区互相隔离且写入不能越界；
- LangMem 每个候选只能引用通过 Lead 验证且与自身声明匹配的 Evidence；无匹配项保持 `model_inference/candidate`；
- LangMem 候选只能通过 Lead 的 Policy Gate 持久化。

### 13.4 可选真实测试

通过显式环境变量启用跨进程 A2A 和真实 OTLP Collector 测试，默认核心测试不依赖外部服务。

## 14. 完成标准

- Researcher、Builder、Reviewer 均能作为独立 HTTP A2A 服务启动。
- Lead 能发现、认证、调用和恢复三个远程角色任务。
- 每个角色使用独立绝对工作区，文件工具不依赖 CWD 且无法越界。
- 同一业务 Attempt 的传输重试和 Lead 重启不会重复执行 Builder。
- Researcher/Reviewer 能按规则降级，Builder 不会静默本地重做。
- Artifact 下载经过 Origin、大小、MIME 和 SHA-256 校验。
- Tool、MCP、Evidence 和权限数据仍通过现有控制面。
- OpenInference Trace 能写入 JSONL并可选导出 OTLP。
- W3C Trace Context 能跨 Lead 与角色 HTTP 进程保持 Trace ID。
- LangMem 候选必须经过现有 Memory Governance 才能持久化。
- 未配置 A2A、LangMem 或 OTLP 时，现有本地和离线模式保持兼容。
- 原有核心测试全部通过，并新增 A2A、Tracing 和 LangMem 的单元与集成测试。

## 15. 实施拆分

为了让每个阶段都可独立测试和回滚，实施计划应拆成以下顺序：

1. 配置模型、依赖锁定和 TraceProvider 基础。
2. OpenInference/OpenTelemetry Span 接入和 JSONL 降级。
3. `MemberExecutor` 抽象与本地实现迁移。
4. A2A Registry、Client、Artifact Transport 和恢复映射。
5. 三个角色 A2A Service Host。
6. TeamRuntime 远程路由、重试和角色降级。
7. LangMem Candidate Extractor 与 Memory Governance 接口。
8. CLI、健康检查、文档和跨进程验收。

每一步必须先新增失败测试，再写最小实现，并保持本地 Team Mode 可运行。
