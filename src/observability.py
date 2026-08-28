from __future__ import annotations

import json
import secrets
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping


_CURRENT: ContextVar[tuple["_Span", ...]] = ContextVar("trace_spans", default=())
_REMOTE: ContextVar[tuple[str, str] | None] = ContextVar("remote_trace", default=None)
_SECRET_KEYS = {"authorization", "api_key", "apikey", "token", "password", "secret", "openai_api_key"}


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: "[REDACTED]" if str(k).lower() in _SECRET_KEYS else redact_sensitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(v) for v in value]
    if isinstance(value, tuple):
        return [redact_sensitive(v) for v in value]
    return value


@dataclass(frozen=True)
class TraceIdentifiers:
    trace_id: str
    span_id: str
    parent_span_id: str | None = None


class _Span:
    def __init__(self, provider: "JsonlTraceProvider", name: str, attributes: dict[str, Any] | None, identifiers: TraceIdentifiers):
        self.provider, self.name, self.attributes, self.identifiers = provider, name, redact_sensitive(attributes or {}), identifiers
        self.started = datetime.now(timezone.utc)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = redact_sensitive(value)

    def record_exception(self, exc: BaseException) -> None:
        self.attributes["exception.type"] = type(exc).__name__
        self.attributes["exception.message"] = str(exc)[:500]


class JsonlTraceProvider:
    def __init__(self, path: str | Path, service_name: str = "workflow-agent") -> None:
        self.path = Path(path)
        self.service_name = service_name
        self._lock = threading.Lock()

    @contextmanager
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[_Span]:
        parent = _CURRENT.get()[-1] if _CURRENT.get() else None
        remote = _REMOTE.get()
        trace_id = parent.identifiers.trace_id if parent else (remote[0] if remote else secrets.token_hex(16))
        parent_id = parent.identifiers.span_id if parent else (remote[1] if remote else None)
        span = _Span(self, name, attributes, TraceIdentifiers(trace_id, secrets.token_hex(8), parent_id))
        token = _CURRENT.set((*_CURRENT.get(), span))
        try:
            yield span
        except BaseException as exc:
            span.record_exception(exc)
            raise
        finally:
            _CURRENT.reset(token)
            record = {"name": span.name, "service": self.service_name, "trace_id": span.identifiers.trace_id, "span_id": span.identifiers.span_id, "parent_span_id": span.identifiers.parent_span_id, "attributes": redact_sensitive(span.attributes), "ended_at": datetime.now(timezone.utc).isoformat()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def inject(self, carrier: MutableMapping[str, str]) -> None:
        current = _CURRENT.get()[-1] if _CURRENT.get() else None
        if current:
            carrier["traceparent"] = f"00-{current.identifiers.trace_id}-{current.identifiers.span_id}-01"

    @contextmanager
    def extract(self, carrier: Mapping[str, str]) -> Iterator[None]:
        value = carrier.get("traceparent", "")
        parts = value.split("-")
        valid = len(parts) == 4 and len(parts[1]) == 32 and len(parts[2]) == 16 and all(c in "0123456789abcdef" for c in parts[1].lower() + parts[2].lower())
        token = _REMOTE.set((parts[1], parts[2]) if valid else None)
        try:
            yield
        finally:
            _REMOTE.reset(token)


class NoOpTraceProvider(JsonlTraceProvider):
    def __init__(self) -> None:
        pass

    @contextmanager
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[_Span]:
        parent = _CURRENT.get()[-1] if _CURRENT.get() else None
        span = _Span(self, name, attributes, TraceIdentifiers(parent.identifiers.trace_id if parent else secrets.token_hex(16), secrets.token_hex(8), parent.identifiers.span_id if parent else None))
        token = _CURRENT.set((*_CURRENT.get(), span))
        try:
            yield span
        finally:
            _CURRENT.reset(token)

    def inject(self, carrier: MutableMapping[str, str]) -> None:
        JsonlTraceProvider.inject(self, carrier)

    @contextmanager
    def extract(self, carrier: Mapping[str, str]) -> Iterator[None]:
        yield


def build_trace_provider(config, output_dir: str = "outputs"):
    if not getattr(config, "enabled", False):
        return NoOpTraceProvider()
    return JsonlTraceProvider(getattr(config, "jsonl_fallback", f"{output_dir}/traces/agent.jsonl"), getattr(config, "service_name", "workflow-agent"))
