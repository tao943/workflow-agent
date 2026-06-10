import json

from src.events import Event


def test_event_serializes_to_json():
    payload = json.loads(Event("step.started", "sess_1", {"step": 1}).to_json())

    assert payload["type"] == "step.started"
    assert payload["session_id"] == "sess_1"
    assert payload["data"]["step"] == 1
