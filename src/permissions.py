import fnmatch
import sys
from dataclasses import dataclass
from typing import Literal

from src.config import PermissionRule
from src.storage import Storage


PermissionAction = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PermissionDecision:
    action: PermissionAction
    permission: str
    pattern: str
    reason: str


def evaluate_permission(permission: str, pattern: str, rules: list[PermissionRule]) -> PermissionDecision:
    matched = [rule for rule in rules if fnmatch.fnmatch(permission, rule.permission) and fnmatch.fnmatch(pattern, rule.pattern)]
    if not matched:
        return PermissionDecision("ask", permission, pattern, "没有匹配规则，默认询问。")

    for rule in reversed(matched):
        if rule.action == "deny":
            return PermissionDecision("deny", permission, pattern, "匹配 deny 规则。")
    for rule in reversed(matched):
        if rule.action == "allow":
            return PermissionDecision("allow", permission, pattern, "匹配 allow 规则。")
    return PermissionDecision("ask", permission, pattern, "匹配 ask 规则。")


def request_permission(
    session_id: str,
    storage: Storage,
    decision: PermissionDecision,
    description: str,
    auto_approve: bool,
    output_format: str = "default",
) -> bool:
    if decision.action == "allow":
        storage.add_permission(session_id, decision.permission, decision.pattern, "allow", scope="always", source="config")
        return True

    if decision.action == "deny":
        storage.add_permission(session_id, decision.permission, decision.pattern, "deny", scope="always", source="config")
        return False

    if auto_approve:
        storage.add_permission(session_id, decision.permission, decision.pattern, "allow", scope="once", source="user")
        return True

    storage.add_permission(session_id, decision.permission, decision.pattern, "ask", scope="once", source="runtime")
    if output_format == "json":
        return False
    if not sys.stdin.isatty():
        return False

    answer = input(f"Agent 请求权限 {decision.permission}:{decision.pattern}（{description}），是否允许？y/n: ")
    return answer.strip().lower() in {"y", "yes"}
