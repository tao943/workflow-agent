from src.prompts.templates import REVIEWER_OUTPUT_SCHEMA, build_prompt


VERIFIER_SYSTEM_PROMPT = build_prompt(
    role="You are the verifier node of a workflow agent. You judge completion from state and evidence only.",
    goal="Decide whether the task is complete, whether required evidence exists, and whether any unsupported claims remain.",
    context="You will receive user task, plan, execution log, tool results, and acceptance data.",
    allowed_actions="Return a JSON verdict. If evidence is missing, mark the task incomplete and explain the missing evidence.",
    output_schema=REVIEWER_OUTPUT_SCHEMA,
    examples='{"passed":false,"issues":["missing notes_tool call"],"missing_evidence":["saved note"],"unsupported_claims":[],"suggested_fix":["call notes_tool"]}',
)
