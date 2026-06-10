PROMPT_VERSION = "prompt-layer.v2"

FORBIDDEN_CLAIMS = """Forbidden Claims:
- Do not claim you scanned, listed, read, searched, saved, retrieved, or wrote anything unless a matching tool call or evidence card is present.
- Do not place unsupported facts in verified findings.
- If a required tool failed or is missing, report the gap explicitly.
- Do not output hidden chain-of-thought. Output only the requested schema.
"""

EVIDENCE_RULES = """Evidence Rules:
- Treat evidence cards as the only source of verified facts.
- Every verified claim must cite one or more evidence_ids.
- Claims without evidence_ids must be placed under assumptions or open_questions.
- Tool failures must remain failures; never rewrite them as success.
"""

PROMPT_SECTIONS = """Use this structure:
Role
Goal
Available Context
Allowed Actions
Forbidden Claims
Output Schema
Examples
"""

PLANNER_OUTPUT_SCHEMA = """Output Schema:
Return valid JSON only:
{
  "steps": [
    {
      "description": "short executable step",
      "tool_name": null,
      "tool_args": null,
      "required_evidence": [],
      "done_criteria": "observable completion rule"
    }
  ]
}
"""

TEAM_LEAD_OUTPUT_SCHEMA = """Output Schema:
Return valid JSON only:
{
  "intent": "analysis|planning|research|build|review|mixed",
  "confidence": 0.0,
  "rationale_summary": "short routing reason, no hidden chain-of-thought",
  "required_capabilities": [],
  "candidates": [
    {"strategy": "evidence-first", "tasks": []}
  ],
  "suggested_graph": {
    "tasks": [
      {
        "id": "short_stable_id",
        "title": "task title",
        "description": "task instructions",
        "assigned_to": "candidate member name",
        "depends_on": [],
        "node_type": "task",
        "optional": false,
        "condition": "always",
        "required_tools": [],
        "required_evidence": [],
        "acceptance_criteria": "observable completion rule",
        "risk_level": "low",
        "max_retries": 0,
        "replan_policy": "never",
        "why_this_member": "short reason",
        "why_now": "short reason",
        "skip_condition": ""
      }
    ]
  }
}
"""

MEMBER_OUTPUT_SCHEMA = """Output Schema:
Return valid JSON only:
{
  "findings": [{"claim": "...", "evidence_ids": ["ev_1"]}],
  "assumptions": [],
  "risks": [],
  "next_steps": []
}
"""

REVIEWER_OUTPUT_SCHEMA = """Output Schema:
Return valid JSON only:
{
  "passed": false,
  "issues": [],
  "missing_evidence": [],
  "unsupported_claims": [],
  "suggested_fix": []
}
"""

SUMMARIZER_OUTPUT_SCHEMA = """Output Format:
Use exactly these Markdown sections:
## 已证实结论
## 合理推断
## 未验证假设
## 下一步
"""


def build_prompt(role: str, goal: str, context: str, allowed_actions: str, output_schema: str, examples: str = "") -> str:
    return "\n\n".join(
        [
            f"Prompt Version: {PROMPT_VERSION}",
            f"Role\n{role}",
            f"Goal\n{goal}",
            f"Available Context\n{context}",
            f"Allowed Actions\n{allowed_actions}",
            FORBIDDEN_CLAIMS,
            EVIDENCE_RULES,
            output_schema,
            f"Examples\n{examples or 'None'}",
        ]
    )
