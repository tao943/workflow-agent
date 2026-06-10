from __future__ import annotations

from typing import Any


class SessionRepository:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def list(self) -> list[dict]:
        return self.storage.list_sessions()

    def create(self, session_id: str, title: str, agent: str, parent_id: str | None = None) -> None:
        self.storage.create_session(session_id, title, agent, parent_id)

    def update_status(self, session_id: str, status: str) -> None:
        self.storage.update_session_status(session_id, status)


class EventRepository:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def add(self, session_id: str, event_type: str, data: dict) -> None:
        self.storage.add_event(session_id, event_type, data)

    def list(self, session_id: str) -> list[dict]:
        return self.storage.list_events(session_id)


class TeamRepository:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def get_run(self, team_run_id: str) -> dict | None:
        return self.storage.get_team_run(team_run_id)

    def list_tasks(self, team_run_id: str) -> list[dict]:
        return self.storage.list_team_tasks(team_run_id)

    def list_members(self, team_run_id: str) -> list[dict]:
        return self.storage.list_team_members(team_run_id)


class ContextRepository:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def get_snapshot(self, snapshot_id: str) -> dict | None:
        return self.storage.get_context_snapshot(snapshot_id)

    def list_feedback(self, session_id: str) -> list[dict]:
        return self.storage.list_context_feedback(session_id)


class PermissionRepository:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def list(self) -> list[dict]:
        return self.storage.list_permissions()

    def revoke(self, permission_id: int) -> bool:
        return self.storage.revoke_permission(permission_id)

