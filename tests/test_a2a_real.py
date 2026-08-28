import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_role(role: str, port: int, token: str, workspace: Path) -> subprocess.Popen:
    workspace.mkdir(parents=True, exist_ok=True)
    script = ("import uvicorn; from src.config import AppConfig, A2AConfig, A2ARemoteAgentConfig; from src.a2a_service import build_role_app; "
              f"cfg=AppConfig(a2a=A2AConfig(enabled=True, agents={{'{role}': A2ARemoteAgentConfig(enabled=True, token_env='ROLE_TOKEN', workspace_root=r'{workspace}')}})); "
              f"uvicorn.run(build_role_app(cfg, '{role}', 'http://127.0.0.1:{port}'), host='127.0.0.1', port={port}, log_level='error')")
    env = os.environ.copy()
    env["ROLE_TOKEN"] = token
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    return subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _wait_ready(port: int, process: subprocess.Popen) -> None:
    deadline = time.time() + 15
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(process.stderr.read().decode(errors="replace"))
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.3).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise TimeoutError(f"A2A role on port {port} did not become ready")


@pytest.mark.a2a
def test_real_a2a_three_independent_role_processes(tmp_path):
    if os.getenv("RUN_REAL_A2A") != "1":
        pytest.skip("Set RUN_REAL_A2A=1 to run real A2A process tests.")
    roles = [("researcher", "research-secret"), ("builder", "builder-secret"), ("reviewer", "review-secret")]
    processes, ports = [], []
    try:
        for role, token in roles:
            port = _free_port(); ports.append(port)
            processes.append(_start_role(role, port, token, tmp_path / role))
        for port, process in zip(ports, processes):
            _wait_ready(port, process)
        for (role, token), port in zip(roles, ports):
            base = f"http://127.0.0.1:{port}"
            card = httpx.get(base + "/.well-known/agent-card.json", timeout=2)
            assert card.status_code == 200
            assert card.json()["skills"][0]["id"] == f"workflow-agent-{role}"
            assert httpx.post(base + "/a2a/rest/v1/message:send", json={}, timeout=2).status_code == 401
            response = httpx.post(base + "/a2a/rest/v1/message:send", headers={"Authorization": f"Bearer {token}"}, json={"role": role, "task": "health-check"}, timeout=2)
            assert response.status_code == 200
            assert response.json()["status"] == "completed"
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)


@pytest.mark.observability
def test_real_otlp_http_export_receives_span():
    if os.getenv("RUN_REAL_OTLP") != "1":
        pytest.skip("Set RUN_REAL_OTLP=1 to run real OTLP collector tests.")
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from src.observability import OpenTelemetryTraceProvider
    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers.get("content-length", "0")))))
            self.send_response(200); self.end_headers()
        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        provider = OpenTelemetryTraceProvider("outputs/test-otlp.jsonl", otlp_endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces")
        with provider.start_span("integration.test"):
            pass
        provider._tracer_provider.force_flush()
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.1)
        assert received and received[0][0] == "/v1/traces" and received[0][1]
    finally:
        server.shutdown(); server.server_close()
