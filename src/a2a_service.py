from __future__ import annotations

import hmac
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from src.observability import NoOpTraceProvider


def build_role_agent_card(role: str, base_url: str) -> dict:
    return {"name": f"workflow-agent-{role}", "description": f"Workflow {role} agent", "url": base_url.rstrip("/"), "version": "1.0", "skills": [{"id": f"workflow-agent-{role}", "name": role}], "capabilities": {"streaming": False}, "supported_interfaces": [{"url": base_url.rstrip("/") + "/a2a/rest", "protocol_binding": "HTTP+JSON"}]}


def build_role_app(config, role: str, base_url: str, runtime_factory=None, trace_provider=None) -> FastAPI:
    app = FastAPI(title=f"workflow-agent-{role}")
    trace_provider = trace_provider or NoOpTraceProvider()
    token_env = config.a2a.agents.get(role).token_env if role in config.a2a.agents else ""

    @app.get("/.well-known/agent-card.json")
    def card():
        return build_role_agent_card(role, base_url)

    @app.get("/health")
    def health():
        return {"status": "ok", "role": role}

    @app.post("/a2a/rest/v1/message:send")
    async def send(payload: dict, authorization: str | None = Header(default=None), traceparent: str | None = Header(default=None), tracestate: str | None = Header(default=None)):
        with trace_provider.extract({"traceparent": traceparent or "", "tracestate": tracestate or ""}):
            with trace_provider.start_span("a2a.remote", {"a2a.role": role}):
                expected = os.getenv(token_env, "")
                supplied = authorization or ""
                if not expected or not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], expected):
                    raise HTTPException(status_code=401, detail="unauthorized")
                if runtime_factory is not None:
                    result = runtime_factory(role, payload)
                    return JSONResponse({"status": "completed", "result": result})
                return {"status": "completed", "role": role}

    return app


def serve_role(config, role: str, host: str, port: int) -> None:
    import uvicorn
    uvicorn.run(build_role_app(config, role, f"http://{host}:{port}"), host=host, port=port)
