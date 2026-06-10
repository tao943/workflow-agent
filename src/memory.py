import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.state import MAX_MEMORY_ITEMS, MemoryEntry


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
MEMORY_FILE = OUTPUT_DIR / "memory.json"
MEMORY_DB = OUTPUT_DIR / "memory.sqlite"
RECENT_MEMORY_ITEMS = 5
RELEVANT_MEMORY_ITEMS = 5
MEMORY_PROMPT_CHAR_LIMIT = 5000
PROJECT_PROFILE_ID = "mem_project_profile"
PROJECT_PROFILE_TASK = "项目长期记忆"


def load_memory(db_path: str | Path | None = None) -> list[MemoryEntry]:
    resolved_db = Path(db_path) if db_path else MEMORY_DB
    if resolved_db.exists():
        return evolve_memory(_load_memory_from_sqlite(resolved_db))[-MAX_MEMORY_ITEMS:]

    if not MEMORY_FILE.exists():
        return []

    try:
        payload = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    memory: list[MemoryEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        entry = _normalize_entry(item)
        if entry:
            memory.append(entry)

    evolved = evolve_memory(memory)[-MAX_MEMORY_ITEMS:]
    save_memory(evolved, resolved_db)
    return evolved


def save_memory(memory: list[MemoryEntry], db_path: str | Path | None = None) -> None:
    resolved_db = Path(db_path) if db_path else MEMORY_DB
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_memory_to_sqlite(memory[-MAX_MEMORY_ITEMS:], resolved_db)
    MEMORY_FILE.write_text(
        json.dumps(memory[-MAX_MEMORY_ITEMS:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def remember_interaction(
    memory: list[MemoryEntry],
    user_task: str,
    final_answer: str,
    tool_results: list[dict] | None = None,
) -> list[MemoryEntry]:
    next_memory = list(memory)
    next_memory.append(
        _new_interaction_entry(user_task, final_answer, tool_results or [])
    )
    return evolve_memory(next_memory)[-MAX_MEMORY_ITEMS:]


def evolve_memory(memory: list[MemoryEntry]) -> list[MemoryEntry]:
    interactions = [
        item for item in memory
        if item.get("id") != PROJECT_PROFILE_ID and item.get("user_task") != PROJECT_PROFILE_TASK
    ][-(MAX_MEMORY_ITEMS - 1):]
    if not interactions:
        return []

    profile = _build_project_profile(interactions)
    return (interactions + [profile])[-MAX_MEMORY_ITEMS:]


def select_memory_for_task(
    user_task: str,
    memory: list[MemoryEntry],
    recent_limit: int = RECENT_MEMORY_ITEMS,
    relevant_limit: int = RELEVANT_MEMORY_ITEMS,
) -> list[MemoryEntry]:
    if not memory:
        return []

    recent = list(memory[-recent_limit:])
    recent_ids = {item.get("id", id(item)) for item in recent}
    candidates = [item for item in memory if item.get("id", id(item)) not in recent_ids]
    query_terms = _keywords(user_task)

    scored: list[tuple[int, int, MemoryEntry]] = []
    for index, item in enumerate(candidates):
        score = _score_memory(query_terms, item)
        if score > 0:
            scored.append((score, index, item))

    relevant = [item for _, _, item in sorted(scored, key=lambda row: (-row[0], row[1]))[:relevant_limit]]
    selected = relevant + recent
    seen: set[str] = set()
    deduped: list[MemoryEntry] = []
    for item in selected:
        key = str(item.get("id") or item.get("user_task") or id(item))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_memory_for_prompt(user_task: str, memory: list[MemoryEntry]) -> str:
    selected = select_memory_for_task(user_task, memory)
    if not selected:
        return "无"

    lines = ["以下是经过压缩和筛选的项目记忆，只使用仍然相关的事实："]
    for index, item in enumerate(selected, start=1):
        lines.append(f"\n[{index}] 历史任务：{item.get('user_task', '')}")
        summary = item.get("summary") or item.get("final_answer", "")
        if summary:
            lines.append(f"摘要：{_clip(str(summary), 500)}")
        for label, key in (("事实", "facts"), ("偏好", "preferences"), ("决策", "decisions")):
            values = item.get(key, [])
            if values:
                lines.append(f"{label}：")
                lines.extend(f"- {_clip(str(value), 220)}" for value in values[:5])
        tools = item.get("tools_used", [])
        if tools:
            lines.append(f"工具：{', '.join(str(tool) for tool in tools[:6])}")

    text = "\n".join(lines)
    return _clip(text, MEMORY_PROMPT_CHAR_LIMIT)


def _load_memory_from_sqlite(db_path: Path) -> list[MemoryEntry]:
    _init_memory_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, user_task, summary, facts_json, preferences_json,
                   decisions_json, tools_used_json, keywords_json, final_answer
            FROM memory_entries
            WHERE active = 1
            ORDER BY position ASC, created_at ASC
            """
        ).fetchall()
    memory = []
    for row in rows:
        entry = _normalize_entry(
            {
                "id": row[0],
                "created_at": row[1],
                "user_task": row[2],
                "summary": row[3],
                "facts": _json_list(row[4]),
                "preferences": _json_list(row[5]),
                "decisions": _json_list(row[6]),
                "tools_used": _json_list(row[7]),
                "keywords": _json_list(row[8]),
                "final_answer": row[9],
            }
        )
        if entry:
            memory.append(entry)
    return memory


def _save_memory_to_sqlite(memory: list[MemoryEntry], db_path: Path) -> None:
    _init_memory_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE memory_entries SET active = 0")
        for position, item in enumerate(memory):
            conn.execute(
                """
                INSERT INTO memory_entries (
                    id, position, created_at, user_task, summary, facts_json,
                    preferences_json, decisions_json, tools_used_json, keywords_json, final_answer, active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    position = excluded.position,
                    created_at = excluded.created_at,
                    user_task = excluded.user_task,
                    summary = excluded.summary,
                    facts_json = excluded.facts_json,
                    preferences_json = excluded.preferences_json,
                    decisions_json = excluded.decisions_json,
                    tools_used_json = excluded.tools_used_json,
                    keywords_json = excluded.keywords_json,
                    final_answer = excluded.final_answer,
                    active = 1
                """,
                (
                    item.get("id") or f"mem_{uuid4().hex[:12]}",
                    position,
                    item.get("created_at", ""),
                    item.get("user_task", ""),
                    item.get("summary", ""),
                    json.dumps(_string_list(item.get("facts")), ensure_ascii=False),
                    json.dumps(_string_list(item.get("preferences")), ensure_ascii=False),
                    json.dumps(_string_list(item.get("decisions")), ensure_ascii=False),
                    json.dumps(_string_list(item.get("tools_used")), ensure_ascii=False),
                    json.dumps(_string_list(item.get("keywords")), ensure_ascii=False),
                    item.get("final_answer", ""),
                ),
            )


def _init_memory_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                position INTEGER NOT NULL,
                created_at TEXT,
                user_task TEXT NOT NULL,
                summary TEXT,
                facts_json TEXT NOT NULL,
                preferences_json TEXT NOT NULL,
                decisions_json TEXT NOT NULL,
                tools_used_json TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                final_answer TEXT
            )
            """
        )
        columns = [row[1] for row in conn.execute("PRAGMA table_info(memory_entries)").fetchall()]
        if "active" not in columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN active INTEGER NOT NULL DEFAULT 1")


def _new_interaction_entry(user_task: str, final_answer: str, tool_results: list[dict]) -> MemoryEntry:
    return {
        "id": f"mem_{uuid4().hex[:12]}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_task": user_task,
        "summary": _compact_summary(final_answer),
        "facts": _extract_facts(final_answer),
        "preferences": _extract_preferences(user_task, final_answer),
        "decisions": _extract_decisions(final_answer),
        "tools_used": _tools_used(tool_results),
        "keywords": sorted(_keywords(user_task) | _keywords(final_answer)),
        "final_answer": final_answer[:2000],
    }


def _build_project_profile(memory: list[MemoryEntry]) -> MemoryEntry:
    preferences = _dedupe_preserve_order(
        value
        for item in memory
        for value in _string_list(item.get("preferences"))
    )[:12]
    decisions = _dedupe_preserve_order(
        value
        for item in memory
        for value in _string_list(item.get("decisions"))
    )[:16]
    facts = _dedupe_preserve_order(
        value
        for item in memory
        for value in _string_list(item.get("facts"))
    )[:16]
    tools = _dedupe_preserve_order(
        value
        for item in memory
        for value in _string_list(item.get("tools_used"))
    )[:16]
    keywords = sorted(
        {"长期记忆", "项目记忆", "memory", "上下文"}
        | set(tools)
        | set(term for item in memory for term in _string_list(item.get("keywords")))
    )
    summary_parts = [
        f"已沉淀 {len(memory)} 条历史交互。",
        f"稳定偏好 {len(preferences)} 条。",
        f"关键决策 {len(decisions)} 条。",
        f"项目事实 {len(facts)} 条。",
    ]
    return {
        "id": PROJECT_PROFILE_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_task": PROJECT_PROFILE_TASK,
        "summary": " ".join(summary_parts),
        "facts": facts,
        "preferences": preferences,
        "decisions": decisions,
        "tools_used": tools,
        "keywords": keywords,
        "final_answer": "",
    }


def _normalize_entry(item: dict) -> MemoryEntry | None:
    user_task = str(item.get("user_task", "")).strip()
    final_answer = str(item.get("final_answer", "")).strip()
    summary = str(item.get("summary", "")).strip() or _compact_summary(final_answer)
    if not user_task or not (final_answer or summary):
        return None
    return {
        "id": str(item.get("id") or f"mem_{uuid4().hex[:12]}"),
        "created_at": str(item.get("created_at") or ""),
        "user_task": user_task,
        "summary": summary,
        "facts": _string_list(item.get("facts")) or _extract_facts(final_answer),
        "preferences": _string_list(item.get("preferences")) or _extract_preferences(user_task, final_answer),
        "decisions": _string_list(item.get("decisions")) or _extract_decisions(final_answer),
        "tools_used": _string_list(item.get("tools_used")),
        "keywords": _string_list(item.get("keywords")) or sorted(_keywords(user_task) | _keywords(summary)),
        "final_answer": final_answer[:2000],
    }


def _score_memory(query_terms: set[str], item: MemoryEntry) -> int:
    if not query_terms:
        return 0
    keywords = set(_string_list(item.get("keywords")))
    text = " ".join(
        [
            str(item.get("user_task", "")),
            str(item.get("summary", "")),
            " ".join(_string_list(item.get("facts"))),
            " ".join(_string_list(item.get("decisions"))),
        ]
    ).lower()
    overlap = len(query_terms & keywords)
    substring_hits = sum(1 for term in query_terms if term in text)
    return overlap * 3 + substring_hits


def _compact_summary(text: str, limit: int = 800) -> str:
    clean = " ".join(str(text).split())
    if not clean:
        return ""
    return _clip(clean, limit)


def _extract_facts(text: str) -> list[str]:
    return _extract_relevant_lines(
        text,
        ("已", "新增", "修改", "优化", "修复", "测试", "通过", "失败", "工具", "模式", "记忆", "权限", "session", "checkpoint"),
    )


def _extract_preferences(user_task: str, final_answer: str) -> list[str]:
    return _extract_relevant_lines(
        "\n".join([user_task, final_answer]),
        ("以后", "每次", "默认", "希望", "喜欢", "不要", "需要", "优先", "阶段性", "git", "commit"),
    )


def _extract_decisions(text: str) -> list[str]:
    return _extract_relevant_lines(
        text,
        ("决定", "采用", "改成", "升级", "不再", "只允许", "禁止", "保留", "过滤", "约束", "提交"),
    )


def _extract_relevant_lines(text: str, keywords: tuple[str, ...], limit: int = 8) -> list[str]:
    results: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip(" -\t")
        if not line or len(line) < 4:
            continue
        if any(keyword.lower() in line.lower() for keyword in keywords):
            results.append(_clip(line, 260))
        if len(results) >= limit:
            break
    return results


def _tools_used(tool_results: list[dict]) -> list[str]:
    tools: list[str] = []
    for item in tool_results:
        tool = str(item.get("tool", "")).strip()
        if tool and tool not in tools:
            tools.append(tool)
    return tools


def _dedupe_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _keywords(text: str) -> set[str]:
    lowered = str(text).lower()
    ascii_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lowered))
    chinese_terms = {
        word
        for word in (
            "memory",
            "记忆",
            "上下文",
            "工具",
            "权限",
            "计划",
            "执行",
            "检查",
            "总结",
            "会话",
            "session",
            "checkpoint",
            "断点",
            "恢复",
            "rag",
            "opencode",
            "profile",
            "skill",
            "git",
            "提交",
            "模型",
            "速度",
            "循环",
        )
        if word in lowered
    }
    return ascii_terms | chinese_terms


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return _string_list(payload)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def format_memory_for_prompt(user_task: str, memory: list[MemoryEntry]) -> str:
    selected = select_memory_for_task(user_task, memory)
    if not selected:
        return "无"

    lines = ["以下是按相关性筛选出的项目记忆，只使用仍然可靠的事实："]
    for index, item in enumerate(selected, start=1):
        lines.append(f"\n[{index}] 历史任务：{item.get('user_task', '')}")
        summary = item.get("summary") or item.get("final_answer", "")
        if summary:
            lines.append(f"摘要：{_clip(str(summary), 500)}")
        for label, key in (("事实", "facts"), ("偏好", "preferences"), ("决策", "decisions")):
            values = item.get(key, [])
            if values:
                lines.append(f"{label}：")
                lines.extend(f"- {_clip(str(value), 220)}" for value in values[:5])
        tools = item.get("tools_used", [])
        if tools:
            lines.append(f"工具：{', '.join(str(tool) for tool in tools[:6])}")

    return _clip("\n".join(lines), MEMORY_PROMPT_CHAR_LIMIT)


def _extract_preferences(user_task: str, final_answer: str) -> list[str]:
    return _extract_relevant_lines(
        "\n".join([user_task, final_answer]),
        ("以后", "每次", "默认", "希望", "喜欢", "不要", "需要", "优先", "阶段性", "git", "commit"),
    )


def _extract_facts(text: str) -> list[str]:
    return _extract_relevant_lines(
        text,
        ("已", "新增", "修改", "优化", "修复", "测试", "通过", "失败", "工具", "模式", "记忆", "权限", "session", "checkpoint"),
    )


def _extract_decisions(text: str) -> list[str]:
    return _extract_relevant_lines(
        text,
        ("决定", "采用", "改成", "升级", "不再", "只允许", "禁止", "保留", "过滤", "约束", "提交"),
    )


def _keywords(text: str) -> set[str]:
    lowered = str(text).lower()
    ascii_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lowered))
    chinese_terms = {
        word
        for word in (
            "记忆",
            "上下文",
            "工具",
            "权限",
            "计划",
            "执行",
            "检查",
            "总结",
            "会话",
            "断点",
            "恢复",
            "提交",
            "模型",
            "速度",
            "循环",
            "评测",
            "测试",
        )
        if word in lowered
    }
    return ascii_terms | chinese_terms
