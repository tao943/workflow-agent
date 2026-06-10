import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str
    session_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class EventSink:
    def __init__(self, session_id: str, output_format: str = "default") -> None:
        self.session_id = session_id
        self.output_format = output_format
        self.events: list[Event] = []

    def emit(self, event_type: str, **data: Any) -> Event:
        event = Event(type=event_type, session_id=self.session_id, data=data)
        self.events.append(event)
        if self.output_format == "json":
            print(event.to_json())
        return event
