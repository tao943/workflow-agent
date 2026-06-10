import shutil
from pathlib import Path
from uuid import uuid4

from src.project_files import ProjectFileFilter


def test_project_file_filter_excludes_runtime_noise() -> None:
    workspace = (Path("outputs") / "test_tmp" / f"project_files_{uuid4().hex}").resolve()
    try:
        (workspace / "src").mkdir(parents=True)
        (workspace / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "main.pyc").write_text("compiled", encoding="utf-8")
        (workspace / ".uv-cache").mkdir()
        (workspace / ".uv-cache" / "cache.txt").write_text("cache", encoding="utf-8")
        (workspace / "outputs" / "context").mkdir(parents=True)
        (workspace / "outputs" / "context" / "artifact.txt").write_text("large", encoding="utf-8")

        items = [item.relative_to(workspace).as_posix() for item in ProjectFileFilter(workspace).iter_project_files(workspace)]

        assert items == ["src/main.py"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
