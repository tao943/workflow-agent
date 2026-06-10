from src.config import AppConfig
from src.runtime import AgentRuntime, RuntimeOptions
from src.storage import Storage
from pathlib import Path
from uuid import uuid4


def test_agent_runtime_runs_and_records_result():
    root = Path("outputs") / "test_runtime" / uuid4().hex[:8]
    config = AppConfig(output_dir=str(root / "outputs"), checkpoint_path=str(root / "checkpoints.sqlite"))
    storage = Storage(root / "agent.sqlite")
    runtime = AgentRuntime(config, llm=None, storage=storage)

    result = runtime.run(RuntimeOptions(task="测试 runtime", no_memory=True))

    assert result.status == "completed"
    assert result.run_id
    assert result.session_id
    assert result.final_answer
    assert result.events


def test_agent_runtime_build_context():
    root = Path("outputs") / "test_runtime" / uuid4().hex[:8]
    config = AppConfig(output_dir=str(root / "outputs"), checkpoint_path=str(root / "checkpoints.sqlite"))
    storage = Storage(root / "agent.sqlite")
    storage.create_session("sess_ctx", "ctx", "build")
    storage.add_message("msg_1", "sess_ctx", "user", "hello context")
    runtime = AgentRuntime(config, llm=None, storage=storage)

    bundle = runtime.build_context("context", "sess_ctx")

    assert "最近消息" in bundle.system_context
    assert bundle.estimated_chars > 0
