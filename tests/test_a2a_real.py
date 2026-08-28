import os
import pytest


@pytest.mark.a2a
def test_real_a2a_processes_are_opt_in():
    if os.getenv("RUN_REAL_A2A") != "1":
        pytest.skip("Set RUN_REAL_A2A=1 to run real A2A process tests.")
    pytest.xfail("Real three-process orchestration is environment-specific and must be configured with service workspaces.")


@pytest.mark.observability
def test_real_otlp_is_opt_in():
    if os.getenv("RUN_REAL_OTLP") != "1":
        pytest.skip("Set RUN_REAL_OTLP=1 to run real OTLP collector tests.")
    pytest.xfail("Requires an explicitly configured OTLP collector endpoint.")
