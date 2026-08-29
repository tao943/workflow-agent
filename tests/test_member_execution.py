from src.member_execution import MemberExecutionResult, MemberTaskRequest


def test_member_request_has_stable_execution_identity_and_workspace(tmp_path):
    request = MemberTaskRequest(team_run_id="run", task_id="task", logical_task_id="build", member_name="builder", role="builder", profile="build", title="t", description="d", required_tools=[], required_evidence=[], acceptance_criteria="", session_id="s", auto_approve=True, output_format="default", no_memory=True, workspace_root=str(tmp_path), execution_id="exec-1", idempotency_key="run:build:0")
    assert request.execution_id == "exec-1"
    assert request.workspace_root == str(tmp_path)


def test_member_result_defaults_are_serializable():
    result = MemberExecutionResult(member_name="researcher", role="researcher", status="completed", final_answer="ok", acceptance={}, tool_calls=[], artifacts=[], events=[], execution_mode="local")
    assert result.remote_task_id == ""
