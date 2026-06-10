from __future__ import annotations

import os
import time
from typing import Any

from src.tools.registry import ToolContext, _rag_search_execute


def rag_health(query: str = "health check") -> dict[str, Any]:
    service_base = os.getenv("RAG_API_BASE", "").rstrip("/")
    started = time.perf_counter()
    if not service_base and not os.path.exists("outputs/knowledge_base.json"):
        return {
            "ok": False,
            "code": "not_configured",
            "service_base": service_base,
            "knowledge_base_id": os.getenv("RAG_KNOWLEDGE_BASE_ID", "kb_001"),
            "latency_ms": 0,
            "message": "RAG is not configured. Set RAG_API_BASE or add outputs/knowledge_base.json.",
        }

    result = _rag_search_execute(
        {"query": query, "top_k": 1, "mode": os.getenv("RAG_SEARCH_MODE", "retrieve")},
        ToolContext(session_id="rag_health", step_id="rag_health", output_dir="outputs"),
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    metadata = result.metadata or {}
    code = str(metadata.get("code") or ("ok" if result.status == "success" else "error"))
    sources = metadata.get("sources") if isinstance(metadata.get("sources"), list) else []
    diagnostics = metadata.get("diagnostics") if isinstance(metadata.get("diagnostics"), dict) else {}
    return {
        "ok": result.status == "success" and code not in {"not_configured", "service_unavailable"},
        "status": result.status,
        "code": code,
        "service_base": service_base,
        "knowledge_base_id": os.getenv("RAG_KNOWLEDGE_BASE_ID", "kb_001"),
        "search_mode": os.getenv("RAG_SEARCH_MODE", "retrieve"),
        "latency_ms": latency_ms,
        "source_count": len(sources),
        "sources": sources,
        "diagnostics": diagnostics,
        "error": result.error or "",
        "message": result.display_output or result.output or result.error or "",
    }
