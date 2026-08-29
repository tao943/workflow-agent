# LangMem Production Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing LangMem adapter to the real Team/A2A execution path so validated role results become governed memories and active role memories are recalled on later tasks.

**Architecture:** `TeamRuntime` remains the owner of memory orchestration. LangMem's official `create_memory_manager` only converts successful member reports into `MemoryCandidate` objects; `MemoryGovernanceRuntime.evaluate_candidates` remains the only persistence and activation path. Retrieval is scoped to `project:team:<role>` and injected as bounded, explicitly fallible context into both local and remote member tasks.

**Tech Stack:** Python 3.13, LangMem 0.0.30, LangChain/LangGraph, Pydantic 2, SQLite FTS, pytest.

## Global Constraints

- Use `create_memory_manager`; never use `create_memory_store_manager` or let LangMem write `memory.sqlite`.
- Extract only from completed member results whose Evidence Gate passed.
- A member is eligible only when `evidence_check.passed is True`; a missing value is never treated as success, including the A2A branch.
- Lead assigns and persists stable evidence IDs/source refs before member execution. Remote text cannot introduce accepted evidence.
- Keep only candidate-declared Evidence IDs and source refs that intersect that member's accepted evidence.
- Unsupported candidates remain `model_inference` with confidence at most `0.4`.
- LangMem failure is non-critical; emit `memory_extraction_error` and use the existing rule consolidator only when `fallback_to_rule_consolidator` is true.
- Preserve legacy behavior when `langmem.enabled` is false or `--no-memory` is used.
- Never put raw tool output, secrets, bearer tokens, or full memory contents in events.

---

### Task 1: Make the adapter compatible with official LangMem results

**Files:**
- Modify: `src/langmem_runtime.py`
- Modify: `tests/test_langmem_runtime.py`

**Interfaces:**
- Consumes: official `ExtractedMemory(id: str, content: BaseModel)` values returned by `MemoryManager.invoke`.
- Produces: `LangMemCandidateExtractor.extract(...) -> list[MemoryCandidate]` with verified per-candidate attribution.

- [ ] **Step 1: Write failing tests for official result unwrapping and bounded attribution**

Add a named-tuple result whose `content` is `ExtractedAgentMemory`, assert it converts successfully, and assert unaccepted IDs/refs are removed and confidence is capped. Add a factory-recording test that confirms the official manager receives the schema, insert/update enabled, delete disabled, and evidence-aware instructions.

- [ ] **Step 2: Run the adapter tests and verify RED**

Run: `& 'D:\桌面\工作流agent\.venv\Scripts\python.exe' -m pytest tests/test_langmem_runtime.py -q`

Expected: failure when Pydantic attempts to validate the official `(id, content)` tuple or when instructions are absent.

- [ ] **Step 3: Implement minimal official-shape normalization**

Normalize official named tuples and compatible wrappers before Pydantic validation. Pass bounded instructions listing only allowed Evidence IDs/source refs. Validate the role namespace and avoid mutating caller data.

- [ ] **Step 4: Run the adapter tests and verify GREEN**

Run the command from Step 2. Expected: all adapter tests pass.

---

### Task 2: Persist LangMem candidates through TeamRuntime governance

**Files:**
- Modify: `src/team.py`
- Create: `tests/test_team_langmem.py`

**Interfaces:**
- Produces: `TeamRuntime._consolidate_team_memory(...)` and a constructor injection point for a test/production extractor factory.
- Consumes: member payload fields `role`, `status`, `final_answer`, `evidence`, `evidence_check`, and `task_metadata`.

- [ ] **Step 1: Write a failing successful-extraction integration test**

Run an offline Team workflow with `langmem.enabled=true` and an injected extractor. Assert it is called only when `status == "completed"` and `evidence_check.passed is True`, receives stable per-member evidence IDs/source refs, and its candidates are persisted through `MemoryPolicyGate` under `project:team:<role>`.

- [ ] **Step 2: Run the focused Team test and verify RED**

Run: `& 'D:\桌面\工作流agent\.venv\Scripts\python.exe' -m pytest tests/test_team_langmem.py -q`

Expected: failure because `TeamRuntime` never invokes LangMem.

- [ ] **Step 3: Implement extraction, validation, and centralized persistence**

Before member execution, assign each Lead-collected evidence item `ev_<sha256(team_run_id, task_id, tool, canonical args, artifact identity)[:16]>` and `team:<team_run_id>:task:<task_id>:evidence:<id>`, persist both in the existing team evidence message, and reuse them regardless of list rendering order. After synthesis, collect eligible member results, build the accepted catalog only from those persisted Lead-issued values, extract all candidates before writing any, and call `MemoryGovernanceRuntime.evaluate_candidates` once. Ignore IDs claimed only by remote output. Keep the legacy compact interaction entry for compatibility, but do not also run rule consolidation after successful LangMem extraction. Add tests for evidence reordering, cross-member ID separation, and a forged remote ID.

- [ ] **Step 4: Write and verify RED for extraction failure policy**

Add tests for `memory_extraction_error`, rule fallback enabled, and rule fallback disabled. The fallback must process only the same eligible results, once per role, in `project:team:<role>`; pass successful acceptance to the rule consolidator only when that role has Lead-accepted evidence. Ensure event data contains role/error class only, not result content or secrets.

- [ ] **Step 5: Implement the non-critical fallback policy and verify GREEN**

On any extraction exception, emit the sanitized event and either call the existing rule consolidator once or continue without governed consolidation according to configuration.

---

### Task 3: Recall governed role memory in local and A2A tasks

**Files:**
- Modify: `src/team.py`
- Modify: `src/memory_runtime.py`
- Modify: `src/config.py`
- Modify: `tests/test_team_langmem.py`
- Modify: `README.md`
- Modify: `agent_config.json.example`

**Interfaces:**
- Produces: bounded role-memory context appended to member task descriptions.
- Consumes: `MemoryGovernanceRuntime.retrieve(query, node, namespace, limit, session_id)`.

- [ ] **Step 1: Write failing local and remote recall tests**

Seed one active memory in `project:team:researcher` and another in `project:team:builder`. Assert the Researcher receives only its own memory in a local run and in a monkeypatched A2A HTTP request. Assert both `no_memory=true` and `langmem.enabled=false` perform neither extraction nor role-memory retrieval. Cover all supported memory types for each role.

- [ ] **Step 2: Run recall tests and verify RED**

Run: `& 'D:\桌面\工作流agent\.venv\Scripts\python.exe' -m pytest tests/test_team_langmem.py -k 'recall or disabled or no_memory' -q`

Expected: failure because TeamRuntime does not retrieve role namespaces.

- [ ] **Step 3: Implement bounded role-scoped recall**

Add explicit `team_researcher`, `team_builder`, and `team_reviewer` retrieval type mappings. Retrieve only active memories in `project:team:<role>`, cap by `memory.retrieval_limit` and `langmem.recall_char_limit` (default 4000), strip control characters, render IDs/subjects/clipped contents plus trust labels inside `BEGIN_UNTRUSTED_GOVERNED_MEMORY`/`END_UNTRUSTED_GOVERNED_MEMORY`, and state that this block is evidence-bound context rather than instructions. Append the same bounded context before both A2A submission and local `AgentRuntime` execution.

- [ ] **Step 4: Document activation and operational behavior**

Document `langmem.enabled`, the requirement for a configured real LLM, fallback behavior, role namespaces, model-call cost, and `--no-memory`. Keep the example default disabled for backwards compatibility while showing how to enable it.

- [ ] **Step 5: Run focused and full verification**

Run:

```powershell
& 'D:\桌面\工作流agent\.venv\Scripts\python.exe' -m pytest tests/test_langmem_runtime.py tests/test_memory_runtime.py tests/test_team_mode.py tests/test_team_langmem.py tests/test_a2a_runtime.py tests/test_a2a_service.py -q
& 'D:\桌面\工作流agent\.venv\Scripts\python.exe' -m pytest -q
& 'D:\桌面\工作流agent\.venv\Scripts\python.exe' -c "from langmem import create_memory_manager; print('langmem-import-ok')"
```

Expected: zero failures and `langmem-import-ok`.
