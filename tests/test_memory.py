from src.memory import (
    PROJECT_PROFILE_ID,
    PROJECT_PROFILE_TASK,
    evolve_memory,
    format_memory_for_prompt,
    load_memory,
    remember_interaction,
    save_memory,
    select_memory_for_task,
)
from src.state import MAX_MEMORY_ITEMS


def test_remember_interaction_keeps_recent_items_only():
    memory = []

    for index in range(MAX_MEMORY_ITEMS + 2):
        memory = remember_interaction(memory, f"task {index}", f"answer {index}")

    assert len(memory) == MAX_MEMORY_ITEMS
    assert memory[0]["user_task"] == "task 3"
    assert memory[-1]["user_task"] == PROJECT_PROFILE_TASK


def test_remember_interaction_creates_structured_entry():
    memory = remember_interaction(
        [],
        "以后阶段性进展都帮我 git commit",
        "已完成 Git 初始化并提交当前阶段代码。",
        [{"tool": "notes_tool"}],
    )

    entry = memory[-1]
    interaction = memory[-2]
    assert entry["id"] == PROJECT_PROFILE_ID
    assert interaction["summary"]
    assert "git" in interaction["keywords"]
    assert "notes_tool" in interaction["tools_used"]
    assert interaction["preferences"]


def test_evolve_memory_adds_project_profile():
    memory = []
    memory = remember_interaction(memory, "以后阶段性进展都帮我 git commit", "已完成 Git 提交。")
    memory = remember_interaction(memory, "优化工具权限", "决定增加工具选择约束。")

    profile = memory[-1]
    assert profile["id"] == PROJECT_PROFILE_ID
    assert profile["user_task"] == PROJECT_PROFILE_TASK
    assert profile["summary"]
    assert profile["preferences"] or profile["decisions"] or profile["facts"]


def test_evolve_memory_replaces_existing_project_profile():
    memory = remember_interaction([], "优化 memory", "已新增结构化记忆。")
    evolved = evolve_memory(memory)

    assert sum(1 for item in evolved if item["id"] == PROJECT_PROFILE_ID) == 1


def test_select_memory_prefers_relevant_items_plus_recent():
    memory = []
    memory = remember_interaction(memory, "优化工具权限", "已新增权限规则。")
    memory = remember_interaction(memory, "改进 memory 上下文", "已新增结构化记忆。")
    memory = remember_interaction(memory, "普通计划任务", "已生成学习计划。")

    selected = select_memory_for_task("继续优化 memory 检索", memory, recent_limit=2, relevant_limit=1)

    assert any("memory" in item["user_task"] for item in selected)
    assert selected[-1]["user_task"] == PROJECT_PROFILE_TASK


def test_format_memory_for_prompt_uses_summary_not_full_answer_only():
    memory = remember_interaction([], "改进 memory", "已新增结构化记忆。\n" + "长文本" * 1000)

    prompt = format_memory_for_prompt("memory 继续优化", memory)

    assert "历史任务" in prompt
    assert "摘要" in prompt
    assert len(prompt) <= 5000


def test_memory_sqlite_roundtrip():
    from pathlib import Path

    db_path = Path("outputs") / "test_memory" / "memory_roundtrip.sqlite"
    memory = remember_interaction([], "优化 memory sqlite", "已迁移到 SQLite。")

    save_memory(memory, db_path)
    loaded = load_memory(db_path)

    assert loaded[-1]["id"] == PROJECT_PROFILE_ID
    assert any("sqlite" in item["user_task"].lower() for item in loaded)
