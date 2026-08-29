# A2A SWE-bench 90+ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the five selected Requests SWE-bench cases through an independent HTTP A2A Builder and iteratively improve real test pass rates to at least 90% per case.

**Architecture:** The coordinator creates one isolated Requests checkout per case and sends the unchanged problem statement to an independent Builder process through the project's official A2A HTTP+JSON client path. The Builder uses a scoped patch-edit capability, returns a patch/artifact, and the coordinator submits that patch to the official SWE-bench 5.0.2 Docker grader. A local test-pass percentage is reported alongside the official binary resolved result.

**Tech Stack:** Existing A2A SDK/ FastAPI service, OpenTelemetry/OpenInference, Python, Git, Docker, `swebench==5.0.2`.

## Global Constraints

- Cases: `psf__requests-1963`, `psf__requests-2317`, `psf__requests-2674`, `psf__requests-3362`, `psf__requests-863`.
- A2A transport must be real independent HTTP process, not an in-process mock.
- Builder workspace is isolated per case and cannot escape its root.
- Never use Gold Patch or test patch during generation; only official grader metadata is used for evaluation.
- Preserve official resolved/unresolved semantics; never rewrite scores.
- 90% target means `(FAIL_TO_PASS passed + PASS_TO_PASS passed) / (FAIL_TO_PASS + PASS_TO_PASS) >= 0.90` for each case.
- Record A2A task ID, execution ID, patch hash, test counts, resolved result, and OTLP trace IDs.
- A2A SDK/API: use installed `a2a-sdk==1.1.2`, canonical Agent Card `supportedInterfaces` HTTP+JSON binding, and `message/send` task payload; verify with an SDK client↔server integration test.
- The server binds to an allowlisted workspace at process startup; it ignores/rejects client-supplied workspace roots, symlink escapes, and cross-case paths.
- Iterations use a new `business_attempt` and execution ID while preserving idempotency within each attempt.
- The 90% metric is a separate explicit test ratio; official SWE-bench `resolved` remains binary and is never inferred from the ratio.
- OTLP requires Collector readiness, exporter flush, per-case client/server trace IDs, and secret/prompt-content checks.
- Keep concurrency at 1 and stop before free disk drops below 8 GiB.

---

### Task 1: Add a scoped patch-edit capability

**Files:**
- Modify: `src/config.py`
- Modify: `src/tools/registry.py`
- Modify: `src/assignment.py`
- Test: `tests/test_tool_protocol.py`

Add a `apply_patch` tool that accepts a unified diff, validates every path is inside the active workspace, rejects runtime/credential paths, applies with `git apply --check` before applying, and records provenance. Expose it only to Builder when explicitly enabled by benchmark config; keep existing write-file semantics unchanged.

### Task 2: Make the independent A2A Builder benchmark-capable

**Files:**
- Modify: `src/a2a_service.py`
- Modify: `src/a2a_runtime.py`
- Modify: `src/team.py`
- Test: `tests/test_a2a_real.py`

Pass the scoped workspace and benchmark task metadata through A2A HTTP+JSON, execute Builder in its own process, return patch/artifact metadata, preserve execution identity across retries, and emit `a2a.remote` plus OpenInference spans. Builder must return an explicit error instead of silently falling back to local execution.

### Task 3: Build the A2A SWE-bench runner

**Files:**
- Create: `outputs/swebench/run_a2a_generation.py`
- Create: `outputs/swebench/a2a-runtime-config.json`
- Test: `outputs/swebench/test_a2a_generation.py`

Start one Builder HTTP process per case, send the exact problem statement through the official A2A client path, capture returned patch, verify it applies to the recorded base commit, and save complete per-case metadata. Load model credentials only from the existing environment without printing them.

### Task 4: Iterate against official tests

**Files:**
- Create: `outputs/swebench/a2a-iterations/`
- Create: `outputs/swebench/a2a-summary.json`
- Create: `outputs/swebench/a2a-summary.md`

For each case, run official Docker grading, parse FAIL_TO_PASS/PASS_TO_PASS counts, and if below 90%, send the failing test output and current patch context to a fresh A2A Builder attempt in the same case checkout. Cap each case at 5 iterations and retain every attempt; stop only when all five meet 90% or the cap is exhausted, then report the exact blocker.

### Task 5: Verify final evidence

Run the project's full test suite, A2A real-process tests, patch-tool tests, and the official grader reports. Confirm every case has an A2A execution identity, OTLP evidence, no secret text, and a denominator-preserving result.
