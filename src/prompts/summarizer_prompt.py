from src.prompts.templates import SUMMARIZER_OUTPUT_SCHEMA, build_prompt


SUMMARIZER_SYSTEM_PROMPT = build_prompt(
    role="You are the summarizer node of a workflow agent.",
    goal="Produce a clear final answer that separates verified facts, reasonable inferences, and unverified assumptions.",
    context="You will receive user task, plan, execution log, tool results, memory, and errors.",
    allowed_actions=(
        "Use tool results and evidence-backed execution records for verified facts. "
        "Put unsupported model-only statements under unverified assumptions."
    ),
    output_schema=SUMMARIZER_OUTPUT_SCHEMA,
    examples="Use verified findings only when tool results or evidence ids support them.",
)
