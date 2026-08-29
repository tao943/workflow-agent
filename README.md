# Workflow Agent Runtime

一个基于 **Python + LangGraph** 的工程化 Workflow Agent 项目。它从教学型「规划 -> 执行 -> 检查 -> 总结」流程逐步演进为一个可审计、可扩展、可测评的 **Agent Runtime**，支持单 Agent、Team Mode、多工具来源、RAG、MCP、Skill Plugin、Context Harness、长期记忆和 Benchmark 评估。

项目定位不是“万能聊天机器人”，而是一个用于学习和实践 Agent 工程化能力的 CLI Runtime：

```text
User Task
  -> Planner / Router
  -> Executor
  -> ToolRuntime
  -> Evidence / Permission / Provenance Gate
  -> Verifier / Reviewer
  -> Synthesizer
```

## 核心能力

- **单 Agent 工作流**：基于 LangGraph 实现 Planner、Executor、Verifier、Summarizer 节点，支持结构化 PlanStep、最大轮数限制和离线 fallback。
- **多 Agent Team Mode**：支持 Lead Agent 调度 Planner、Researcher、Builder、Reviewer，使用任务图 / DAG 执行、成员上下文隔离、Reviewer 验收和 Lead Synthesizer 汇总。
- **统一工具运行时**：通过 `ToolRuntime` 和 `ToolProvider` 统一管理 built-in tools、RAG、MCP、Skill Plugin、Claw 动态工具。
- **权限与安全边界**：支持 `allow / ask / deny` 权限规则，写文件、RAG、MCP、Skill、Docker 等高风险能力必须经过权限系统。
- **工具幻觉治理**：通过 Evidence Gate 和 Argument Provenance Gate 校验工具证据和关键参数来源，避免模型凭空声明“已读取、已保存、已检索”。
- **Context Harness**：支持长工具结果卸载、artifact 索引、结构化 summary、上下文缓存、反馈监督和 Team 成员上下文隔离。
- **Memory Governance**：支持 semantic、episodic、procedural、preference 四类长期记忆，使用 SQLite FTS5 / BM25 召回，并保留 trust、confidence、candidate / active / invalidated 状态。
- **RAG / MCP / Skill 扩展**：支持真实 RAG 接入、MCP fetch / search 类工具、项目内 Skill Package，以及 Python subprocess / Docker 沙箱执行。
- **Eval 与 Benchmark**：内置本地 Eval Harness、DeepEval 风格指标适配、Claw-Eval workflow，覆盖工具正确性、参数正确性、答案证据覆盖和任务完成率。

## 项目结构

```text
workflow-agent/
  src/
    main.py                 # CLI 入口
    runtime.py              # AgentRuntime 执行内核
    team.py                 # Team Mode runtime
    assignment.py           # 任务路由、任务图和动态重规划
    tool_runtime.py          # 工具执行、权限、证据、截断和事件
    tool_providers.py        # 工具来源统一抽象
    argument_provenance.py   # 参数来源验证
    context_harness.py       # 上下文卸载、反馈、缓存、分支
    memory_runtime.py        # 长期记忆治理
    mcp_runtime.py           # MCP client runtime
    skill_plugins.py         # 可执行 Skill runtime
    claw_workflow.py         # Claw-Eval workflow adapter
    answer_synthesis.py      # 证据驱动答案生成
    nodes/                   # planner / executor / verifier / summarizer
    tools/                   # built-in tools and registry
  tests/                     # pytest 单元、集成和回归测试
  evals/benchmarks/          # 本地 benchmark cases
  skills/                    # 项目内 Skill Package
  scripts/                   # 辅助脚本
```

## 快速开始

### 1. 创建环境

```powershell
python -m venv .venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
```

如果没有 `uv`，也可以使用你自己的 Python 包管理方式安装 `requirements.txt`。

### 2. 配置环境变量

复制示例文件：

```powershell
copy .env.example .env
```

按需配置：

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

也可以直接使用 `--offline` 跑本地 fallback 流程。

## 常用命令

### 单 Agent 运行

```powershell
python -m src.main "帮我制定一个三天学习 LangGraph 的计划，并保存为笔记" --agent build --yes
```

离线模式：

```powershell
python -m src.main "分析当前项目结构并给出优化建议" --offline --no-memory
```

输出 JSON 事件：

```powershell
python -m src.main "计算 12 * (8 + 5)" --offline --format json
```

### Team Mode

```powershell
python -m src.main "分析当前项目结构并给出优化建议" --team default-dev-team --offline --yes --no-memory
```

查看 Team Run：

```powershell
python -m src.main --team-status TEAM_RUN_ID
python -m src.main --team-events TEAM_RUN_ID
python -m src.main --team-tasks TEAM_RUN_ID
python -m src.main --team-messages TEAM_RUN_ID
```

### Session / Run / Context 调试

```powershell
python -m src.main --list-runs
python -m src.main --inspect-run RUN_ID
python -m src.main --inspect-session SESSION_ID
python -m src.main --events SESSION_ID
python -m src.main --show-context "分析当前项目结构"
```

### Memory

```powershell
python -m src.main --list-memory
python -m src.main --memory-stats
python -m src.main --explain-memory-retrieval "LangGraph checkpoint"
python -m src.main "临时任务，不读写记忆" --no-memory
```

### LangMem 生产链路

LangMem 默认关闭，避免在未配置真实模型时产生额外调用和费用。启用方式：

```json
{
  "langmem": {
    "enabled": true,
    "fallback_to_rule_consolidator": true,
    "max_candidates_per_run": 8,
    "recall_char_limit": 4000
  }
}
```

启用后，Lead 只把状态为 completed 且通过 Evidence Gate 的 Researcher、Builder、Reviewer 结果交给 LangMem。LangMem 使用当前运行的真实 LLM 提取候选，不直接写数据库；候选仍必须经过 `MemoryPolicyGate`，并分别写入 `project:team:researcher`、`project:team:builder`、`project:team:reviewer`。Lead 生成的稳定 Evidence ID 才能激活证据记忆，远程 Agent 自报的 Evidence ID 不会被信任。

后续任务只召回当前角色命名空间中的 active 记忆，最多注入 `recall_char_limit` 个字符，并标记为不可信参考上下文。提取失败时，`fallback_to_rule_consolidator=true` 会按角色使用原有规则式 Consolidator；关闭该选项则记录 `memory_extraction_error` 后继续主任务。`--no-memory` 会同时禁止 LangMem 提取和角色记忆召回。

### RAG / MCP / Skill

```powershell
python -m src.main --rag-health --format json
python -m src.main --mcp-health --format json
python -m src.main --list-mcp-tools fetch
python -m src.main --list-skills
python -m src.main --validate-skills
python -m src.main --doctor-sandbox
```

示例 MCP 调用：

```powershell
python -m src.main --run-mcp-tool mcp.fetch.fetch "{\"url\":\"https://example.com\"}" --yes
```

## 测试与评估

### 单元与回归测试

```powershell
python -m pytest -q
```

普通测试默认不依赖真实外部服务。真实集成测试通过环境变量显式开启：

```powershell
$env:RUN_REAL_RAG='1'
$env:RUN_REAL_MCP='1'
$env:RUN_REAL_DOCKER='1'
python -m pytest -m integration
```

### Eval Harness

```powershell
python -m src.main --eval-list
python -m src.main --eval-run smoke --offline --format json
python -m src.main --eval-run team_core --offline --format json
python -m src.main --eval-report EVAL_RUN_ID
```

### Benchmark / DeepEval

```powershell
python -m src.main --benchmark-list
python -m src.main --benchmark-run baseline_v1 --offline
python -m src.main --benchmark-report BENCHMARK_RUN_ID
python -m src.main --benchmark-doctor deepeval
```

### Claw-Eval Workflow

```powershell
python -m src.main --benchmark-doctor claw --offline --format json
python -m src.main --claw-workflow-run tasks/T014_meeting_notes --trials 3 --yes --format json
python -m src.main --claw-workflow-batch claw_smoke_v1 --trials 1 --yes --format json
python -m src.main --claw-workflow-batch claw_smoke_v1 --claw-mode team --team default-dev-team --trials 1 --yes --format json
```

## 工具协议

每个工具通过 `ToolSpec` 描述：

- `name`
- `description`
- `parameters`
- `output_schema`
- `permissions`
- `timeout_seconds`
- `truncate_policy`
- `provenance_policy`
- `argument_provenance_rules`

工具统一返回 `ToolResult`：

```text
status: success | error | denied | invalid_args | timeout
title
output / display_output
metadata
attachments
duration_ms
truncated
raw_output_path
```

## 安全设计

默认安全边界：

- 不支持任意 shell 执行。
- 写文件默认限制在项目内允许目录。
- 高风险工具默认需要权限确认。
- `--yes` 只能批准 `ask`，不能绕过 `deny`。
- 外部网页、MCP 输出、RAG 结果均视为不可信上下文，不能覆盖系统策略。
- Skill 代码默认视为高风险能力，通过 subprocess 或 Docker runtime 受控执行。
- 长工具输出会被卸载为 artifact，避免污染 prompt context。

## Agent Profile

内置 profile：

- `plan`：偏只读规划，不主动执行写入。
- `build`：允许执行构建类任务，写入仍需权限。
- `research`：偏检索与资料收集，适合 RAG / MCP / Web-like 工具。

示例：

```powershell
python -m src.main "制定一个 LangGraph 学习计划" --agent plan
python -m src.main "总结 README 并写入 outputs/summary.md" --agent build --yes
python -m src.main "检索知识库中的 checkpoint 资料" --agent research --yes
```

## 当前重点优化方向

- 更精细的 Team Mode 动态路由与任务图重规划。
- 更稳定的 Answer Synthesizer 与 evidence-aware final answer。
- 更完整的 MCP / Skill / RAG 真实集成测试。
- 与官方 Benchmark 协议更深的对齐。
- 将 Context Harness 与 Memory Governance 进一步收敛为统一控制平面。

## 说明

本项目用于学习和实践 Agent Runtime 工程化，不建议直接用于生产环境。若接入外部 API、MCP Server、Skill 代码或 Docker 沙箱，请先确认权限规则和本地环境安全边界。
# A2A 角色服务

Researcher、Builder、Reviewer 可作为独立 HTTP 进程运行。配置 `agent_config.json` 中的 `a2a.agents.<role>.workspace_root` 必须指向绝对隔离目录；Token 仅通过 `A2A_<ROLE>_TOKEN` 环境变量提供。非回环端点使用 HTTPS，明文 HTTP 仅用于本机开发。

```powershell
python -m src.main --serve-a2a-role researcher --host 127.0.0.1 --port 8101
python -m src.main --serve-a2a-role builder --host 127.0.0.1 --port 8102
python -m src.main --serve-a2a-role reviewer --host 127.0.0.1 --port 8103
python -m src.main --a2a-agents --format json
python -m src.main --doctor-observability --format json
```

Trace 默认写入 `outputs/traces/agent.jsonl`；设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 后可接入 OTLP。LangMem 只生成候选记忆，最终仍由 `MemoryPolicyGate` 决定是否激活。Builder 远程失败不会自动在 Lead 工作区重做。

## 本地 OTLP + Jaeger

开发和集成测试可以使用项目附带的 Docker Compose 观测栈：

```powershell
docker compose -f docker-compose.otel.yml up -d
$env:OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4318/v1/traces"
$env:OTEL_SERVICE_NAME="workflow-agent"
python -m src.main --doctor-observability --format json
```

Jaeger UI 位于 [http://127.0.0.1:16686](http://127.0.0.1:16686)。Collector 接收 OTLP HTTP `4318` 和 gRPC `4317`，再转发到 Jaeger。停止服务：

```powershell
docker compose -f docker-compose.otel.yml down
```

该 Compose 文件用于本地开发/集成测试，不提供生产级持久化、TLS 或认证；生产环境应使用受保护的远程 Collector。
