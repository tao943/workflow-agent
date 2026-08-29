# SWE-bench Lite Five-Case Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an honest, reproducible SWE-bench Lite baseline for five selected `psf/requests` instances using this project's real model/runtime and the official SWE-bench grader.

**Architecture:** Use the official `swebench==5.0.2` package only for dataset metadata, Docker test environments, and grading. Run this project's Builder against an isolated host checkout for each instance, capture the resulting `git diff` as the prediction, and submit those predictions to the official grader. Keep generation failures, empty patches, and grading failures visible in the final report.

**Tech Stack:** Python 3.10+, project CLI, OpenAI-compatible configured model API, Git, Docker Desktop, `swebench==5.0.2` (MIT).

## Global Constraints

- Instances: `psf__requests-1963`, `psf__requests-2317`, `psf__requests-2674`, `psf__requests-3362`, `psf__requests-863`.
- Dataset: `princeton-nlp/SWE-bench_Lite`, `test` split.
- Grader: official SWE-bench 5.0.2 Docker harness.
- Concurrency: `max_workers=1` because Docker has about 7.2 GB RAM and drive D has about 19.4 GB free.
- Dataset revision: resolve and record the Hugging Face commit SHA used by the installed loader; record all five `base_commit` values in metadata.
- Resource limits: allow at most 15 minutes per Agent generation and 90 minutes for official grading; require at least 8 GB free before starting another Docker case.
- Never use or expose gold patches during prediction generation.
- Never print API key values; inherit the existing configured environment only.
- Preserve all existing uncommitted LangMem changes.
- Report empty/error predictions as failures, never silently drop them.
- Enable the project's OpenInference/LangChain instrumentation during generation and record OTLP Collector reachability plus per-case trace export evidence separately from benchmark correctness.

---

### Task 1: Prepare an isolated benchmark environment

**Files:**
- Create: `outputs/swebench/.venv/`
- Create: `outputs/swebench/run-metadata.json`

**Interfaces:**
- Consumes: host Python, Git, Docker Desktop, existing model environment.
- Produces: a Python environment where `python -m swebench.harness.run_evaluation --help` succeeds and metadata records package/Docker versions without secrets.

- [ ] **Step 1: Create a dedicated virtual environment**

Run: `python -m venv outputs/swebench/.venv`

Expected: `outputs/swebench/.venv/Scripts/python.exe` exists.

- [ ] **Step 2: Install the official grader without changing project requirements**

Run: `outputs/swebench/.venv/Scripts/python.exe -m pip install "swebench==5.0.2"`

Expected: command exits 0.

- [ ] **Step 3: Verify the official CLI and Docker**

Run: `outputs/swebench/.venv/Scripts/python.exe -m swebench.harness.run_evaluation --help`

Expected: help includes `--predictions_path`, `--instance_ids`, and `--max_workers`.

Run: `docker version --format '{{.Server.Version}}'`

Expected: a Docker server version is printed.

- [ ] **Step 4: Record non-secret environment metadata**

Write `run-metadata.json` with Python, SWE-bench, Docker, model name, selected IDs, resolved dataset revision, each selected row's `base_commit`, start time, timeouts, disk-free values, OTLP service name, Collector health, and whether an OTLP endpoint is configured. Never store endpoint credentials or API values.

### Task 2: Generate real project predictions

**Files:**
- Create: `outputs/swebench/checkouts/requests/`
- Create: `outputs/swebench/generation/<instance_id>.json`
- Create: `outputs/swebench/predictions.jsonl`

**Interfaces:**
- Consumes: five dataset rows containing `instance_id`, `base_commit`, and `problem_statement`; project CLI at this worktree; existing model configuration.
- Produces: one prediction JSON object per instance with `instance_id`, `model_name_or_path`, and `model_patch`.

- [ ] **Step 1: Fetch only the five public dataset rows at a recorded revision**

Resolve and record the dataset repository revision, then use the Hugging Face dataset API or the installed dataset loader at that revision. Filter by the exact five IDs and validate that exactly five unique rows were returned.

Expected: every row contains a non-empty `base_commit` and `problem_statement`.

- [ ] **Step 2: Clone Requests once and create an isolated checkout per case**

Run a single `git clone https://github.com/psf/requests.git outputs/swebench/checkouts/requests-source`, then create one Git worktree at each row's exact `base_commit`.

Expected: `git rev-parse HEAD` in every case equals the dataset `base_commit`.

- [ ] **Step 3: Run the real project Builder for each case**

From each case checkout, invoke the host project's Python interpreter with `PYTHONPATH=D:\桌面\工作流agent-worktree-a2a` and `python -m src.main`. Pass a benchmark-specific config derived from the current config but with absolute `output_dir`, `checkpoint_path`, and `observability.jsonl_fallback` paths under `outputs/swebench/runtime/<instance_id>` so runtime state never lands in the Requests checkout. Load `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` into the child environment from `D:\桌面\工作流agent\.env` without printing or copying their values. Use `--agent build --yes --no-memory --format json`, the unmodified problem statement, and a 15-minute process timeout. Explicitly instruct the runtime to inspect and edit the current checkout only; do not include `patch` or `test_patch` dataset fields.

Expected: each invocation records exit code, duration, sanitized stdout/stderr paths, selected model name, OTLP service name, JSONL trace path, exported-span evidence when queryable, and whether stdout contains the CLI's explicit fallback marker `模型调用失败，将自动切换到本地 fallback 逻辑` or missing-key marker `未找到 OPENAI_API_KEY`. A timeout or either fallback marker sets `generation_status` to `failed` and yields an empty prediction; it must not be counted as a real-model attempt.

- [ ] **Step 4: Capture complete predictions without repairing them**

Delete or exclude only known benchmark runtime paths, inspect `git status --porcelain`, then run `git add -N -- <explicit untracked source paths>` followed by `git diff --binary --no-ext-diff`. Reject paths outside the checkout and do not stage runtime logs, caches, virtual environments, credentials, or `.env` files. Validate the captured patch with `git apply --check` in a fresh detached worktree at the same `base_commit`.

Expected: append exactly one JSON line per selected ID. Preserve an empty `model_patch` when the project made no valid change, timed out, used fallback, or produced a patch that cannot apply; do not manually fix Agent output.

### Task 3: Grade with the official Docker harness

**Files:**
- Create: `outputs/swebench/grader/`
- Create: `outputs/swebench/official-run.log`

**Interfaces:**
- Consumes: `predictions.jsonl` and the five exact instance IDs.
- Produces: official per-instance test reports and resolved counts.

- [ ] **Step 1: Validate prediction schema and identity set**

Check that the JSONL has exactly five lines, five unique selected IDs, a fixed `model_name_or_path`, and string-valued `model_patch` fields.

Expected: validation exits 0; otherwise stop grading and report the malformed predictions.

- [ ] **Step 2: Check resources before grading**

Require Docker to be healthy and drive D to have at least 8 GB free. If either check fails, mark all not-yet-graded cases as `environment_error`, keep them in the five-case denominator, and stop before risking disk exhaustion.

Expected: Docker is healthy and free space is at least 8 GB.

- [ ] **Step 3: Run official evaluation sequentially**

Run from `outputs/swebench/grader`:

```powershell
..\.venv\Scripts\python.exe -m swebench.harness.run_evaluation `
  --dataset_name princeton-nlp/SWE-bench_Lite `
  --split test `
  --predictions_path ..\predictions.jsonl `
  --instance_ids psf__requests-1963 psf__requests-2317 psf__requests-2674 psf__requests-3362 psf__requests-863 `
  --max_workers 1 `
  --run_id workflow-agent-requests-5
```

Expected: the command completes and official report artifacts exist. If Docker pull/build or an instance test fails, retain its logs and classify it separately from an Agent-resolution failure.

Enforce a 90-minute outer timeout. After each official case artifact appears, re-check free disk; do not start further work below the 8 GB threshold. A timeout, disk threshold breach, or Docker error becomes `environment_error` for affected cases and those cases remain in the denominator.

### Task 4: Verify and report the baseline

**Files:**
- Create: `outputs/swebench/summary.json`
- Create: `outputs/swebench/summary.md`

**Interfaces:**
- Consumes: generation records, predictions, and official grader artifacts.
- Produces: a human-readable and machine-readable result with no inflated scores.

- [ ] **Step 1: Reconcile all five cases**

For every selected ID, record generation status, patch byte count, grader status, resolved/unresolved, FAIL_TO_PASS result, PASS_TO_PASS result, duration, log paths, JSONL trace path, OTLP export status, and Jaeger query link when available.

Expected: exactly five rows; no case is omitted.

- [ ] **Step 2: Calculate the score**

Compute `resolved / 5`. Also report `generated_nonempty_patch / 5` and environment-error count so infrastructure failures are distinguishable.

Expected: score denominator remains five even for empty predictions or generation failures.

- [ ] **Step 3: Validate report claims**

Cross-check every resolved flag against official grader output and every patch size against `predictions.jsonl`. Scan reports for accidental API-key text before presenting them.

Expected: no secret values are present; summary and official artifacts agree.
