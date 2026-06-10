from __future__ import annotations

from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".uv-cache",
    ".uv-tools",
}
DEFAULT_EXCLUDED_OUTPUT_DIRS = {
    "artifacts",
    "checkpoints",
    "context",
    "skills",
}
DEFAULT_EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".sqlite",
    ".db",
}


class ProjectFileFilter:
    def __init__(
        self,
        workspace: Path | None = None,
        excluded_dirs: set[str] | None = None,
        excluded_output_dirs: set[str] | None = None,
        excluded_suffixes: set[str] | None = None,
    ) -> None:
        self.workspace = (workspace or Path.cwd()).resolve()
        self.excluded_dirs = excluded_dirs or DEFAULT_EXCLUDED_DIRS
        self.excluded_output_dirs = excluded_output_dirs or DEFAULT_EXCLUDED_OUTPUT_DIRS
        self.excluded_suffixes = excluded_suffixes or DEFAULT_EXCLUDED_SUFFIXES

    def should_include(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(self.workspace)
        except ValueError:
            return False
        parts = set(relative.parts)
        if parts & self.excluded_dirs:
            return False
        if relative.parts and relative.parts[0] == "outputs":
            if len(relative.parts) == 1 or relative.parts[1] in self.excluded_output_dirs:
                return False
        if resolved.suffix in self.excluded_suffixes:
            return False
        return True

    def iter_project_files(self, root: Path | None = None, limit: int | None = None) -> Iterable[Path]:
        base = (root or self.workspace).resolve()
        yielded = 0
        for item in base.rglob("*"):
            if not item.is_file() or not self.should_include(item):
                continue
            yield item
            yielded += 1
            if limit is not None and yielded >= limit:
                return

