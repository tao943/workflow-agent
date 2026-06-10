import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.argument_provenance import ArgumentProvenanceGate, ArgumentProvenanceResult
from src.config import PermissionRule
from src.context_harness import ContextHarness, ContextHarnessConfig, estimate_tokens
from src.permissions import evaluate_permission, request_permission
from src.session import new_id
from src.storage import Storage
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolResult, execute_tool


@dataclass
class ToolRunRequest:
    name: str
    args: dict[str, Any]
    context: ToolContext
    session_id: str
    step_id: str | None
    enabled_tools: set[str] = field(default_factory=set)
    permission_rules: list[PermissionRule] = field(default_factory=list)
    auto_approve: bool = False
    output_format: str = "default"
    storage: Storage | None = None
    prior_tool_results: list[dict[str, Any]] = field(default_factory=list)


class ToolRuntime:
    def __init__(
        self,
        artifact_dir: str | Path = "outputs/artifacts",
        context_harness: ContextHarness | None = None,
        context_config: ContextHarnessConfig | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.context_harness = context_harness
        self.context_config = context_config or ContextHarnessConfig()

    def run(self, request: ToolRunRequest) -> tuple[ToolResult, list[dict[str, Any]]]:
        approvals: list[dict[str, Any]] = []
        storage = request.storage or Storage()
        if request.name not in TOOL_REGISTRY:
            return self._error(request.name, "Unknown tool.", "invalid_args"), approvals

        tool = TOOL_REGISTRY[request.name]
        if request.enabled_tools and request.name not in request.enabled_tools:
            return self._error(request.name, f"Tool {request.name} is disabled.", "denied"), approvals

        argument_error = self._argument_error(tool.parameters, request.args)
        if argument_error:
            return self._error(
                request.name,
                f"The {request.name} tool was called with invalid arguments. {argument_error}.",
                "invalid_args",
            ), approvals

        for permission in tool.permissions:
            permission_name, pattern = permission.rsplit(":", 1)
            decision = evaluate_permission(permission_name, pattern, request.permission_rules)
            approved = request_permission(
                session_id=request.session_id,
                storage=storage,
                decision=decision,
                description=tool.description,
                auto_approve=request.auto_approve,
                output_format=request.output_format,
            )
            approvals.append(
                {
                    "tool": request.name,
                    "approved": approved,
                    "permission": permission,
                    "decision": decision.action,
                }
            )
            if not approved:
                return self._error(request.name, f"Permission denied for {permission}.", "denied"), approvals

        effective_args = dict(request.args)
        provenance = self._check_argument_provenance(request, effective_args, storage)
        if provenance and not provenance.passed:
            if tool.provenance_policy == "require":
                return self._error(
                    request.name,
                    f"The {request.name} tool was called with unverified arguments. {'; '.join(provenance.issues)}.",
                    "invalid_args",
                ), approvals
            if tool.provenance_policy == "repair":
                if not provenance.repaired_args:
                    return self._error(
                        request.name,
                        f"The {request.name} tool arguments could not be repaired from trusted provenance. {'; '.join(provenance.issues)}.",
                        "invalid_args",
                    ), approvals
                original_args = dict(effective_args)
                effective_args.update(provenance.repaired_args)
                storage.add_event(
                    request.session_id,
                    "tool.argument_provenance.repaired",
                    {
                        "tool": request.name,
                        "step_id": request.step_id,
                        "original_args": original_args,
                        "repaired_args": provenance.repaired_args,
                        "provenance_refs": provenance.provenance_refs,
                        "issues": provenance.issues,
                    },
                )
            elif tool.provenance_policy == "warn":
                storage.add_event(
                    request.session_id,
                    "tool.argument_provenance.failed",
                    {
                        "tool": request.name,
                        "step_id": request.step_id,
                        "issues": provenance.issues,
                        "policy": tool.provenance_policy,
                    },
                )

        storage.add_event(request.session_id, "tool.started", {"tool": request.name, "step_id": request.step_id, "input": effective_args})
        start = time.perf_counter()
        result = execute_tool(request.name, effective_args, request.context)
        duration_ms = int((time.perf_counter() - start) * 1000)
        result = self._finalize_result(request.name, result, tool.truncate_policy.get("max_chars", 4000), duration_ms)
        if provenance:
            result = replace(
                result,
                metadata={
                    **result.metadata,
                    "argument_provenance": {
                        "passed": provenance.passed,
                        "issues": provenance.issues,
                        "repaired_args": provenance.repaired_args,
                        "provenance_refs": provenance.provenance_refs,
                    },
                    "effective_args": effective_args,
                },
            )
        harness = self.context_harness or ContextHarness(
            storage,
            self.context_config,
            Path(request.context.output_dir) / "context",
        )
        artifact = None
        if self.context_config.enabled and result.output and estimate_tokens(result.output) >= self.context_config.offload_threshold_tokens:
            artifact = harness.store.offload(
                result.output,
                {
                    "kind": "tool_result",
                    "summary": result.display_output or result.output[:500],
                    "source_ref": f"{request.session_id}:{request.step_id or ''}:{request.name}",
                    "scope": request.session_id,
                    "trust": "verified" if not result.error else "unverified",
                    "session_id": request.session_id,
                },
            )
            result = replace(
                result,
                raw_output_path=artifact.file_path,
                truncated=True,
                metadata={**result.metadata, "context_artifact_id": artifact.id},
                display_output=f"{artifact.summary}\n\n[full result offloaded to context artifact {artifact.id}]",
            )
        feedback = harness.process_result(
            {
                "type": "tool_result",
                "session_id": request.session_id,
                "tool": request.name,
                "args": effective_args,
                "status": result.status,
                "error": result.error,
                "metadata": result.metadata,
            }
        )
        result = replace(result, metadata={**result.metadata, "feedback_decision": feedback.__dict__})
        storage.add_tool_call(new_id("tool"), request.session_id, request.step_id, request.name, effective_args, result.display_output or result.output, result.error)
        storage.add_event(
            request.session_id,
            "tool.completed",
            {
                "tool": request.name,
                "step_id": request.step_id,
                "status": result.status,
                "output": result.display_output,
                "error": result.error,
                "metadata": result.metadata,
            },
        )
        storage.add_message_part(
            new_id("part"),
            request.step_id or new_id("msg_tool"),
            request.session_id,
            "tool",
            {
                "tool": request.name,
                "input": effective_args,
                "status": result.status,
                "output": result.display_output or result.output,
                "error": result.error,
                "metadata": result.metadata,
            },
        )
        return result, approvals

    def _check_argument_provenance(
        self,
        request: ToolRunRequest,
        args: dict[str, Any],
        storage: Storage,
    ) -> ArgumentProvenanceResult | None:
        tool = TOOL_REGISTRY[request.name]
        if tool.provenance_policy == "off" or not tool.argument_provenance_rules:
            return None
        result = ArgumentProvenanceGate().validate(
            {"tool": request.name, "input": args},
            request.prior_tool_results,
            tool.argument_provenance_rules,
        )
        storage.add_event(
            request.session_id,
            "tool.argument_provenance.checked",
            {
                "tool": request.name,
                "step_id": request.step_id,
                "policy": tool.provenance_policy,
                "passed": result.passed,
                "issues": result.issues,
                "repaired_args": result.repaired_args,
                "provenance_refs": result.provenance_refs,
            },
        )
        return result

    def _argument_error(self, parameters: dict[str, Any], args: dict[str, Any]) -> str:
        if parameters.get("type") == "object" or "properties" in parameters:
            try:
                from jsonschema import ValidationError, validate

                validate(instance=args, schema=parameters)
                return ""
            except ImportError:
                required = parameters.get("required")
                if isinstance(required, list):
                    missing = []
                    for key in required:
                        value = args.get(str(key))
                        if value is None or (isinstance(value, str) and not value.strip()):
                            missing.append(str(key))
                else:
                    missing = self._missing_required_args(parameters.get("properties", {}), args)
                return f"Missing: {', '.join(missing)}" if missing else ""
            except ValidationError as exc:
                return exc.message
        missing = self._missing_required_args(parameters, args)
        return f"Missing: {', '.join(missing)}" if missing else ""

    def _missing_required_args(self, parameters: dict[str, Any], args: dict[str, Any]) -> list[str]:
        missing = []
        for key in parameters:
            if key in {"top_k", "source_filter", "assigned_to", "result", "reason"}:
                continue
            value = args.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(key)
        return missing

    def _finalize_result(self, tool_name: str, result: ToolResult, max_chars: int, duration_ms: int) -> ToolResult:
        output = result.output or ""
        display = output
        raw_output_path = result.raw_output_path
        truncated = result.truncated or bool(result.metadata.get("truncated"))
        if len(output) > max_chars:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact = self.artifact_dir / f"{tool_name}_{int(time.time() * 1000)}.txt"
            artifact.write_text(output, encoding="utf-8")
            display = f"{output[:max_chars]}\n\n[output truncated; raw output saved to {artifact}]"
            raw_output_path = str(artifact)
            truncated = True
        return ToolResult(
            title=result.title,
            output=output,
            metadata=result.metadata,
            error=result.error,
            attachments=result.attachments,
            status=result.status if result.status != "success" else ("error" if result.error else result.status),
            display_output=display,
            raw_output_path=raw_output_path,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def _error(self, tool_name: str, message: str, status: str) -> ToolResult:
        return ToolResult(
            title=f"{tool_name} failed",
            output="",
            error=message,
            status=status,  # type: ignore[arg-type]
            display_output="",
            metadata={"tool": tool_name},
        )
