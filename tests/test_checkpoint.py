from pathlib import Path

from src.checkpoint import load_checkpoint, save_checkpoint
from src.graph import make_initial_state


def test_checkpoint_round_trip(monkeypatch):
    import src.checkpoint as checkpoint

    output_dir = Path("outputs") / "test_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(checkpoint, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(checkpoint, "CHECKPOINT_DB", output_dir / "checkpoints.sqlite")

    state = make_initial_state("测试 checkpoint", auto_approve=True)
    save_checkpoint("test-run", state, "running", "initial")

    loaded = load_checkpoint("test-run")

    assert loaded["user_task"] == "测试 checkpoint"
    assert loaded["auto_approve"] is True


def test_checkpoint_uses_explicit_db_path():
    db_path = Path("outputs") / "test_checkpoints" / "explicit.sqlite"
    state = make_initial_state("测试 explicit checkpoint")

    save_checkpoint("explicit-run", state, "running", "initial", db_path=db_path)
    loaded = load_checkpoint("explicit-run", db_path=db_path)

    assert loaded["user_task"] == "测试 explicit checkpoint"
