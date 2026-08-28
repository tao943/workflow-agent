from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, repr=False)
class ResolvedA2AAgent:
    role: str
    base_origin: str
    agent_card_url: str
    authorization_header: str
    allow_local_fallback: bool
    workspace_root: str

    def __repr__(self) -> str:
        return f"ResolvedA2AAgent(role={self.role!r}, origin={self.base_origin!r})"


class A2ARuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False, details: dict | None = None):
        super().__init__(message)
        self.code, self.retryable, self.details = code, retryable, details or {}


class A2AAgentRegistry:
    def __init__(self, config):
        self.config = config

    def resolve(self, role: str) -> ResolvedA2AAgent:
        agent = self.config.agents.get(role)
        if not self.config.enabled or agent is None or not agent.enabled:
            raise A2ARuntimeError("configuration_error", f"A2A role {role!r} is disabled")
        if role == "builder" and agent.allow_local_fallback:
            raise A2ARuntimeError("configuration_error", "Builder local fallback is forbidden")
        parsed = urlparse(agent.agent_card_url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or not parsed.hostname:
            raise A2ARuntimeError("configuration_error", "invalid A2A endpoint")
        token = os.getenv(agent.token_env) if agent.token_env else None
        if not token:
            raise A2ARuntimeError("configuration_error", "A2A token environment variable is missing")
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return ResolvedA2AAgent(role, origin, agent.agent_card_url, f"Bearer {token}", agent.allow_local_fallback, agent.workspace_root)
