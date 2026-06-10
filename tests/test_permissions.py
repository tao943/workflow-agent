from src.config import PermissionRule
from src.permissions import evaluate_permission


def test_deny_rule_wins_over_allow():
    decision = evaluate_permission(
        "write",
        "outputs/notes.md",
        [
            PermissionRule(permission="write", pattern="*", action="allow"),
            PermissionRule(permission="write", pattern="outputs/*", action="deny"),
        ],
    )

    assert decision.action == "deny"


def test_rag_permission_can_be_blocked():
    decision = evaluate_permission(
        "rag",
        "*",
        [PermissionRule(permission="rag", pattern="*", action="deny")],
    )

    assert decision.action == "deny"
