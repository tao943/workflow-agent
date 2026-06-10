from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


MatchMode = Literal["set_equals", "contains_all", "equals", "in", "path_under"]
FailureAction = Literal["repair", "deny", "warn"]
SourceType = Literal["tool_result", "safe_default", "user_task", "evidence"]


@dataclass
class ArgumentProvenanceRule:
    target_tool: str
    target_arg: str
    source_tool: str
    source_field: str
    match_mode: MatchMode = "equals"
    source_filter: dict[str, Any] = field(default_factory=dict)
    failure_action: FailureAction = "repair"
    source_type: SourceType = "tool_result"
    source_value: Any = None


@dataclass
class ArgumentProvenanceResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    repaired_args: dict[str, Any] = field(default_factory=dict)
    provenance_refs: list[dict[str, Any]] = field(default_factory=list)


class ArgumentProvenanceGate:
    def validate(
        self,
        tool_call: dict[str, Any],
        prior_tool_results: list[dict[str, Any]],
        rules: list[ArgumentProvenanceRule],
    ) -> ArgumentProvenanceResult:
        args = _args_from_tool_call(tool_call)
        applicable = [rule for rule in rules if rule.target_tool == tool_call.get("tool") or rule.target_tool == tool_call.get("tool_name")]
        issues: list[str] = []
        repaired_args: dict[str, Any] = {}
        provenance_refs: list[dict[str, Any]] = []

        for rule in applicable:
            expected = self._extract_source_value(rule, prior_tool_results)
            actual = args.get(rule.target_arg)
            if expected is None:
                issues.append(f"{rule.target_tool}.{rule.target_arg} has no provenance source from {rule.source_tool}.{rule.source_field}")
                continue
            if _matches(actual, expected, rule.match_mode):
                provenance_refs.append(
                    {
                        "target_tool": rule.target_tool,
                        "target_arg": rule.target_arg,
                        "source_tool": rule.source_tool,
                        "source_field": rule.source_field,
                        "value": expected,
                    }
                )
                continue
            issues.append(f"{rule.target_tool}.{rule.target_arg} does not match provenance source {rule.source_tool}.{rule.source_field}")
            repair_value = _repair_value(expected, rule.match_mode)
            if rule.failure_action == "repair" and repair_value is not None:
                repaired_args[rule.target_arg] = repair_value
                provenance_refs.append(
                    {
                        "target_tool": rule.target_tool,
                        "target_arg": rule.target_arg,
                        "source_tool": rule.source_tool,
                        "source_field": rule.source_field,
                        "value": repair_value,
                        "repaired": True,
                    }
                )

        return ArgumentProvenanceResult(not issues, issues, repaired_args, provenance_refs)

    def _extract_source_value(self, rule: ArgumentProvenanceRule, prior_tool_results: list[dict[str, Any]]) -> Any:
        if rule.source_type == "safe_default":
            return rule.source_value
        collected: list[Any] = []
        for item in prior_tool_results:
            if item.get("tool") != rule.source_tool and item.get("tool_name") != rule.source_tool:
                continue
            if item.get("status") != "success":
                continue
            payload = _payload_from_tool_result(item)
            if not _filter_matches(payload, rule.source_filter):
                continue
            value = _get_path(payload, rule.source_field)
            if value is not None:
                collected.append(value)
        return _merge_source_values(collected)


def _args_from_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    for key in ("input", "args", "arguments"):
        value = tool_call.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _payload_from_tool_result(item: dict[str, Any]) -> Any:
    raw = item.get("result")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    response = metadata.get("response_body")
    return response if isinstance(response, dict) else {}


def _filter_matches(payload: Any, filters: dict[str, Any]) -> bool:
    return all(_get_path(payload, key) == value for key, value in filters.items())


def _get_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            values = []
            for item in current:
                if isinstance(item, dict) and part in item:
                    values.append(item[part])
            current = values
        else:
            return None
    return current


def _matches(actual: Any, expected: Any, mode: MatchMode) -> bool:
    if mode == "set_equals":
        return _normalized_set(actual) == _normalized_set(expected)
    if mode == "contains_all":
        return _normalized_set(expected).issubset(_normalized_set(actual))
    if mode == "in":
        return str(actual).strip() in _normalized_set(expected)
    if mode == "path_under":
        actual_text = str(actual or "").strip().replace("\\", "/")
        expected_text = str(expected or "").strip().replace("\\", "/").rstrip("/") + "/"
        return actual_text == expected_text.rstrip("/") or actual_text.startswith(expected_text)
    return actual == expected


def _repair_value(expected: Any, mode: MatchMode) -> Any:
    if mode == "in" and isinstance(expected, list):
        return expected[0] if len(expected) == 1 else None
    return expected


def _merge_source_values(values: list[Any]) -> Any:
    if not values:
        return None
    if all(isinstance(value, list) for value in values):
        merged: list[Any] = []
        seen: set[str] = set()
        for value in values:
            for item in value:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
        return merged
    return values[0]


def _normalized_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if value is None:
        return set()
    return {str(value).strip()} if str(value).strip() else set()
