from src.config import load_config, skills_for_agent
from src.graph import build_graph, make_initial_state
from src.skills import SKILL_REGISTRY, permissions_for_skills, prompt_for_skills, tools_for_skills


def test_skill_registry_groups_tools_and_prompts():
    assert "file_ops" in SKILL_REGISTRY
    assert "read_file" in tools_for_skills(["file_ops"])
    assert "rag_search" in tools_for_skills(["research"])
    assert "RAG" in prompt_for_skills(["research"])


def test_skill_permissions_are_available():
    permissions = permissions_for_skills(["notes"])
    assert any(rule.permission == "write" for rule in permissions)


def test_agent_profile_resolves_skills():
    config = load_config(agent="plan")
    assert "file_ops" in skills_for_agent(config)
    assert "notes" not in skills_for_agent(config)


def test_disabled_skill_prevents_tool_planning():
    app = build_graph(llm=None)
    result = app.invoke(
        make_initial_state(
            "帮我制定一个三天学习 LangGraph 的计划，并保存为笔记",
            enabled_skills=["core"],
            enabled_tools=tools_for_skills(["core"]),
        )
    )

    planned_tools = [step["tool_name"] for step in result["plan"]]
    assert "notes_tool" not in planned_tools
    assert result["plan"][-1]["description"] == "生成完整结果"
