from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.evidence import verify_final_answer_evidence


Confidence = Literal["verified", "inferred", "assumption"]
Priority = Literal["high", "medium", "low"]


@dataclass
class EvidenceCard:
    id: str
    tool: str
    status: str
    summary: str
    path: str = ""
    artifact: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisClaim:
    claim: str
    evidence_ids: list[str]
    confidence: Confidence = "verified"


@dataclass
class Recommendation:
    title: str
    problem: str
    suggestion: str
    priority: Priority
    evidence_ids: list[str]
    risk: str
    next_action: str


@dataclass
class SynthesisResult:
    claims: list[SynthesisClaim]
    problems: list[SynthesisClaim]
    recommendations: list[Recommendation]
    risks: list[str]
    evidence_cards: list[EvidenceCard]
    issues: list[str] = field(default_factory=list)
    analysis_subtype: str = "structure"
    depth: str = "standard"


class LeadSynthesizer:
    def synthesize(self, task: str, team_name: str, member_results: list[dict[str, Any]]) -> SynthesisResult:
        evidence_cards = self.collect_evidence_cards(member_results)
        analysis_subtype, depth = infer_analysis_strategy(task, member_results)
        claims = self.derive_project_facts(evidence_cards, analysis_subtype)
        recommendations = self.generate_recommendations(evidence_cards, analysis_subtype)
        problems = [SynthesisClaim(item.problem, item.evidence_ids, "inferred") for item in recommendations]
        risks = [item.risk for item in recommendations]
        issues = validate_final_answer_contract(claims, recommendations, evidence_cards)
        return SynthesisResult(claims, problems, recommendations, risks, evidence_cards, issues, analysis_subtype, depth)

    def collect_evidence_cards(self, member_results: list[dict[str, Any]]) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        counter = 1
        for result in member_results:
            for item in result.get("evidence") or []:
                cards.append(evidence_card_from_tool_result(counter, item))
                counter += 1
        return cards

    def derive_project_facts(self, evidence_cards: list[EvidenceCard], analysis_subtype: str = "structure") -> list[SynthesisClaim]:
        claims: list[SynthesisClaim] = []
        by_path = {normalize_path(card.path): card for card in evidence_cards if card.path and card.status == "success"}
        list_card = next((card for card in evidence_cards if card.tool == "list_files" and card.status == "success"), None)
        search_cards = [card for card in evidence_cards if card.tool == "search_files" and card.status == "success"]

        if list_card:
            claims.append(SynthesisClaim(f"项目文件列表已收集，当前扫描结果包含 {list_card.metadata.get('count', 'unknown')} 个条目。", [list_card.id]))
        if "readme.md" in by_path:
            claims.append(SynthesisClaim("README 说明该项目是 Python + LangGraph CLI workflow agent。", [by_path["readme.md"].id]))
        if "src/main.py" in by_path:
            claims.append(SynthesisClaim("CLI 入口集中在 src/main.py，包含参数解析、运行入口和 inspect 类命令。", [by_path["src/main.py"].id]))
        if "src/config.py" in by_path:
            claims.append(SynthesisClaim("配置、agent profile、team spec 和权限规则集中在 src/config.py。", [by_path["src/config.py"].id]))
        if "src/runtime.py" in by_path:
            claims.append(SynthesisClaim("单 Agent 编排位于 src/runtime.py，负责 session、context、graph、checkpoint 和 memory 协调。", [by_path["src/runtime.py"].id]))
        if "src/team.py" in by_path:
            claims.append(SynthesisClaim("Team Mode 编排位于 src/team.py，负责团队创建、成员调度、证据收集、验收和最终汇总。", [by_path["src/team.py"].id]))
        if "pytest.ini" in by_path:
            claims.append(SynthesisClaim("pytest 配置已被读取，可用于判断测试 marker、默认运行方式和集成测试策略。", [by_path["pytest.ini"].id]))
        if "src/evals.py" in by_path:
            claims.append(SynthesisClaim("Eval Harness 逻辑位于 src/evals.py，负责本地评测 case、指标和报告。", [by_path["src/evals.py"].id]))
        if "src/claw_workflow.py" in by_path:
            claims.append(SynthesisClaim("Claw-Eval 工作流适配位于 src/claw_workflow.py，负责 trace、官方 grader 和 repair gate。", [by_path["src/claw_workflow.py"].id]))
        for card in search_cards[:3]:
            claims.append(SynthesisClaim(f"源码搜索 `{card.metadata.get('query') or card.metadata.get('pattern') or ''}` 得到 {card.metadata.get('count', 'unknown')} 处匹配。", [card.id]))

        if not claims and evidence_cards:
            successful = [card for card in evidence_cards if card.status == "success"]
            if successful:
                claims.append(SynthesisClaim(f"已收集 {len(successful)} 条成功工具证据，适用于 {analysis_subtype} 分析。", [successful[0].id]))
        return claims

    def generate_recommendations(self, evidence_cards: list[EvidenceCard], analysis_subtype: str = "structure") -> list[Recommendation]:
        recs: list[Recommendation] = []
        by_path = {normalize_path(card.path): card for card in evidence_cards if card.path and card.status == "success"}
        list_card = next((card for card in evidence_cards if card.tool == "list_files" and card.status == "success"), None)
        search_card = next((card for card in evidence_cards if card.tool == "search_files" and card.status == "success"), None)

        def add(title: str, problem: str, suggestion: str, priority: Priority, card: EvidenceCard, risk: str, next_action: str) -> None:
            recs.append(Recommendation(title, problem, suggestion, priority, [card.id], risk, next_action))

        if analysis_subtype in {"structure", "engineering", "architecture"}:
            if "src/main.py" in by_path:
                add("拆分 CLI command handlers", "src/main.py 同时承担参数解析、运行入口和多类 inspect 输出，继续增长会降低维护性。", "将普通运行、Team Mode、benchmark、inspect 类命令拆为独立 command handler，main.py 只保留 argparse 和分发。", "high", by_path["src/main.py"], "拆分时容易改变现有 CLI 行为，需要回归测试保护参数兼容。", "先迁移只读 inspect/health 命令，再迁移运行命令。")
            if "src/team.py" in by_path:
                add("继续拆分 TeamRuntime 边界", "src/team.py 同时处理任务分派、证据采集、成员调度、验收和总结，模块职责偏重。", "将 analysis strategy、evidence planning、member execution、lead synthesis 逐步抽成独立模块。", "high", by_path["src/team.py"], "一次性大拆会增加回归风险，应按子能力逐步迁移。", "优先抽出 analysis strategy 和 evidence planner。")
            if "src/config.py" in by_path:
                add("收敛配置与内置团队定义", "src/config.py 同时包含基础配置、权限规则、agent profile 和 team spec，配置层职责集中。", "将内置 TeamSpec/Profile 移到独立模块或配置文件，同时保留 Pydantic 校验。", "medium", by_path["src/config.py"], "过早拆分会增加初学者理解成本。", "先只抽 TEAM_SPECS，保持 AppConfig 接口不变。")

        if analysis_subtype == "testing":
            card = by_path.get("pytest.ini") or list_card or search_card
            if card:
                add("分层运行真实集成测试", "普通 pytest 已区分核心测试和真实 RAG/MCP/Docker 集成测试，但真实服务测试默认跳过。", "保留快速单测，同时提供明确的 integration/e2e 命令和失败诊断。", "high", card, "真实服务波动容易让日常开发不稳定。", "在 CI 或手动验收中按环境变量开启真实集成测试。")
        if analysis_subtype == "benchmark":
            card = by_path.get("src/claw_workflow.py") or by_path.get("src/evals.py") or search_card or list_card
            if card:
                add("区分 native pass 与 adapter repaired pass", "Benchmark 通过可能来自 adapter 后置修复，而不是 Agent 原生完成任务。", "报告继续拆分 native_tool_success、adapter_repaired、native_pass_rate 和 repaired_pass_rate。", "high", card, "只看总通过率会高估 Agent 能力。", "将 adapter_repaired=true 的 case 自动进入优化队列。")
        if analysis_subtype == "security":
            card = by_path.get("src/permissions.py") or by_path.get("src/tool_runtime.py") or search_card
            if card:
                add("保持权限系统作为确定性安全边界", "联网 MCP、Skill 代码和 Docker sandbox 会扩大外部输入与执行面。", "继续让 ToolRuntime -> Permission -> Evidence 作为唯一工具执行通道。", "high", card, "如果 provider 或 skill 绕过权限层，会破坏安全模型。", "给每类外部工具补充 deny/ask/allow 回归测试。")
        if analysis_subtype == "performance":
            card = by_path.get("src/context_harness.py") or by_path.get("src/runtime.py") or search_card
            if card:
                add("稳定 prompt 前缀并减少无关工具上下文", "多 agent、MCP、Skill 和 Eval 工具增多后，prompt 上下文容易膨胀并降低缓存命中。", "按 profile/task 裁剪工具列表，固定 system/tool schema 顺序，长工具输出走 Context Harness。", "high", card, "过度裁剪工具可能导致 planner 缺少必要能力。", "将 selected tools 和 context bundle 写入 inspect 输出。")

        if list_card and not any(rec.title == "增强最终答案质量回归测试" for rec in recs):
            add("增强最终答案质量回归测试", "最终答案仍可能退化为成员状态、证据日志或过度模板化输出。", "为不同 analysis_subtype 增加章节、证据 id 和 recommendation 数量的弹性断言。", "medium", list_card, "测试过度绑定文案会降低后续表达调整空间。", "断言结构和证据引用，不绑定完整句子。")
        return recs

    def render_final_answer(self, task: str, team_name: str, result: SynthesisResult, member_results: list[dict[str, Any]]) -> str:
        sections = section_titles_for_subtype(result.analysis_subtype)
        lines = [sections[0], ""]
        lines.extend(render_claims(result.claims))

        lines.extend(["", sections[1], ""])
        if result.problems:
            for index, problem in enumerate(result.problems, start=1):
                lines.append(f"{index}. {problem.claim} 证据：{', '.join(problem.evidence_ids)}")
        else:
            lines.append("- 暂未发现有证据支撑的主要问题。")

        lines.extend(["", sections[2], ""])
        if result.recommendations:
            for index, rec in enumerate(result.recommendations, start=1):
                lines.extend([f"{index}. {rec.title}", f"   优先级：{rec.priority}", f"   问题：{rec.problem}", f"   建议：{rec.suggestion}", f"   证据：{', '.join(rec.evidence_ids)}", f"   下一步：{rec.next_action}"])
        else:
            lines.append("- 暂无足够证据生成优化建议。")

        lines.extend(["", sections[3], ""])
        for priority in ("high", "medium", "low"):
            items = [rec for rec in result.recommendations if rec.priority == priority]
            if items:
                evidence_ids = sorted({evidence_id for rec in items for evidence_id in rec.evidence_ids})
                lines.append(f"- {priority}: " + "；".join(rec.title for rec in items) + f" 证据：{', '.join(evidence_ids)}")

        lines.extend(["", sections[4], ""])
        for rec in result.recommendations:
            lines.append(f"- {rec.title}: {rec.risk} 证据：{', '.join(rec.evidence_ids)}")
        if result.issues:
            lines.append("- Evidence gate issues: " + "；".join(result.issues))
        lines.append("- 未验证假设：没有 evidence id 的成员陈述不会进入已证实结论。")
        return "\n".join(lines)

    def render_with_optional_llm(self, task: str, team_name: str, result: SynthesisResult, member_results: list[dict[str, Any]], llm: Any | None = None) -> tuple[str, list[str]]:
        fallback_answer = self.render_final_answer(task, team_name, result, member_results)
        if llm is None:
            return fallback_answer, ["llm_synthesis.skipped"]
        evidence_ids = {card.id for card in result.evidence_cards if card.status == "success"}
        if not evidence_ids:
            return fallback_answer, ["llm_synthesis.skipped:no_successful_evidence"]
        try:
            response = llm.invoke([SystemMessage(content=build_llm_synthesis_system_prompt(result.analysis_subtype)), HumanMessage(content=build_llm_synthesis_user_prompt(task, team_name, result))])
            candidate = str(getattr(response, "content", response) or "").strip()
        except Exception as exc:
            return fallback_answer, [f"llm_synthesis.error:{exc}"]
        gate = verify_final_answer_evidence(candidate, evidence_ids)
        if gate.passed:
            return candidate, ["llm_synthesis.passed"]
        return fallback_answer, ["llm_synthesis.fallback", *gate.issues]


def infer_analysis_strategy(task: str, member_results: list[dict[str, Any]]) -> tuple[str, str]:
    for result in member_results:
        metadata = result.get("task_metadata") or {}
        strategy = metadata.get("analysis_strategy") or metadata.get("evidence_plan") or {}
        if strategy:
            return str(strategy.get("analysis_subtype") or "structure"), str(strategy.get("depth") or "standard")
    text = task.lower()
    if any(keyword in text for keyword in ("测试", "pytest", "coverage")):
        return "testing", "standard"
    if any(keyword in text for keyword in ("benchmark", "deepeval", "claw", "评测", "测评")):
        return "benchmark", "standard"
    if any(keyword in text for keyword in ("安全", "权限", "sandbox", "security")):
        return "security", "standard"
    if any(keyword in text for keyword in ("性能", "速度", "慢", "performance")):
        return "performance", "standard"
    if any(keyword in text for keyword in ("工程化", "runtime", "mcp", "skill", "memory", "context")):
        return "engineering", "standard"
    return "structure", "standard"


def section_titles_for_subtype(subtype: str) -> list[str]:
    if subtype == "testing":
        return ["## 测试体系现状", "## 覆盖缺口", "## 新增测试建议", "## 优先级路线", "## 稳定性风险与下一步"]
    if subtype == "benchmark":
        return ["## Benchmark 现状", "## 失败样例与根因", "## 优化队列", "## 优先级路线", "## 风险与下一步"]
    if subtype == "engineering":
        return ["## 工程化现状", "## 主要风险", "## 改进路线", "## 优先级路线", "## 风险与下一步"]
    if subtype == "security":
        return ["## 安全边界现状", "## 主要风险", "## 安全优化建议", "## 优先级路线", "## 风险与下一步"]
    if subtype == "performance":
        return ["## 性能与上下文现状", "## 主要瓶颈", "## 性能优化建议", "## 优先级路线", "## 风险与下一步"]
    return ["## 当前项目结构分析", "## 主要问题", "## 优化建议", "## 优先级路线", "## 风险与下一步"]


def render_claims(claims: list[SynthesisClaim]) -> list[str]:
    if not claims:
        return ["- 暂未收集到可用于分析的成功证据。"]
    return [f"- {claim.claim} 证据：{', '.join(claim.evidence_ids)}" for claim in claims]


def build_llm_synthesis_system_prompt(subtype: str = "structure") -> str:
    sections = section_titles_for_subtype(subtype)
    return f"""Role
You are the Lead Synthesizer for a multi-agent workflow runtime.

Goal
Turn verified evidence cards into a concise final answer for the user.

Allowed Actions
- Use only the evidence cards, derived claims, and recommendations provided by the runtime.
- You may improve wording, grouping, and prioritization.

Forbidden Claims
- Do not say the project was scanned, read, searched, saved, or written unless a matching evidence id is cited.
- Do not introduce file names, tools, implementation details, or completed work that are not in evidence.
- Do not output member status as the answer.

Output Contract
Use exactly these Markdown sections:
{chr(10).join(sections)}

Evidence Rule
Every factual claim and every recommendation must include an existing evidence id using this format: 证据：ev_1
"""


def build_llm_synthesis_user_prompt(task: str, team_name: str, result: SynthesisResult) -> str:
    lines = [f"User task: {task}", f"Team: {team_name}", f"Analysis subtype: {result.analysis_subtype}", "", "Evidence Cards:"]
    for card in result.evidence_cards:
        lines.extend([f"- {card.id}", f"  tool: {card.tool}", f"  status: {card.status}", f"  summary: {card.summary}", f"  path: {card.path}", f"  artifact: {card.artifact}"])
    lines.extend(["", "Derived Claims:"])
    for claim in result.claims:
        lines.append(f"- {claim.claim} Evidence: {', '.join(claim.evidence_ids)}")
    lines.extend(["", "Runtime Recommendations:"])
    for rec in result.recommendations:
        lines.extend([f"- title: {rec.title}", f"  priority: {rec.priority}", f"  problem: {rec.problem}", f"  suggestion: {rec.suggestion}", f"  risk: {rec.risk}", f"  next_action: {rec.next_action}", f"  evidence: {', '.join(rec.evidence_ids)}"])
    return "\n".join(lines)


def evidence_card_from_tool_result(index: int, item: dict[str, Any]) -> EvidenceCard:
    metadata = dict(item.get("metadata") or {})
    args = dict(item.get("args") or item.get("input") or {})
    path = str(metadata.get("path") or args.get("path") or "")
    tool = str(item.get("tool") or "")
    status = str(item.get("status") or "")
    summary = summarize_tool_evidence(tool, status, path, metadata, args, item)
    return EvidenceCard(f"ev_{index}", tool, status, summary, path, str(item.get("raw_output_path") or ""), metadata)


def summarize_tool_evidence(tool: str, status: str, path: str, metadata: dict[str, Any], args: dict[str, Any], item: dict[str, Any]) -> str:
    if status != "success":
        return f"{tool} failed"
    if tool == "list_files":
        return f"collected project file listing with {metadata.get('count', 'unknown')} entries"
    if tool == "read_file":
        return f"read key file {path or args.get('path')}"
    if tool == "search_files":
        query = args.get("query") or metadata.get("query") or metadata.get("pattern")
        return f"searched files for {query} with {metadata.get('count', 'unknown')} matches"
    return str(item.get("output") or item.get("result") or "")[:240].replace("\n", " ")


def normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    match = re.search(r"(readme\.md|pytest\.ini|evals/[^:]+|tests/[^:]+|src/[^:]+)", normalized)
    return match.group(1) if match else normalized


def validate_final_answer_contract(claims: list[SynthesisClaim], recommendations: list[Recommendation], evidence_cards: list[EvidenceCard]) -> list[str]:
    issues: list[str] = []
    valid_ids = {card.id for card in evidence_cards if card.status == "success"}
    for claim in claims:
        if claim.confidence == "verified" and not claim.evidence_ids:
            issues.append(f"verified claim has no evidence: {claim.claim}")
        for evidence_id in claim.evidence_ids:
            if evidence_id not in valid_ids:
                issues.append(f"claim references missing evidence id: {evidence_id}")
    for rec in recommendations:
        if not rec.evidence_ids:
            issues.append(f"recommendation has no evidence: {rec.title}")
        for evidence_id in rec.evidence_ids:
            if evidence_id not in valid_ids:
                issues.append(f"recommendation references missing evidence id: {evidence_id}")
    return issues
