"""Serializable state shared by physical-design stages and future agents."""

from dataclasses import asdict, dataclass, field
from typing import Any
import json


@dataclass
class DesignState:
    design: str
    stage: str
    tool_versions: dict[str, str] = field(default_factory=dict)
    geometry: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    action_mask: dict[str, list[bool]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
