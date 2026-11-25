"""Physical-design stages and action specifications."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class PhysicalDesignStage(str, Enum):
    FLOORPLANNING = "floorplanning"
    PLACEMENT = "placement"
    CTS = "cts"
    ROUTING = "routing"


STAGE_ORDER: tuple[PhysicalDesignStage, ...] = (
    PhysicalDesignStage.FLOORPLANNING,
    PhysicalDesignStage.PLACEMENT,
    PhysicalDesignStage.CTS,
    PhysicalDesignStage.ROUTING,
)

STAGE_AGENTS: tuple[str, ...] = tuple(stage.value for stage in STAGE_ORDER)


@dataclass(frozen=True)
class StageActionSpec:
    """Declarative OpenROAD parameters for one stage action."""

    name: str
    pre_commands: tuple[str, ...] = ()
    post_commands: tuple[str, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    requires_real_tool_verification: bool = True

    @classmethod
    def from_value(cls, value: Any, *, index: int) -> "StageActionSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, Mapping):
            return cls(
                name=str(value.get("name", f"action-{index}")),
                pre_commands=tuple(str(item) for item in value.get("pre_commands", ())),
                post_commands=tuple(str(item) for item in value.get("post_commands", ())),
                environment={str(k): str(v) for k, v in value.get("environment", {}).items()},
                metadata=dict(value.get("metadata", {})),
                requires_real_tool_verification=bool(value.get("requires_real_tool_verification", True)),
            )
        raise TypeError(f"unsupported stage action specification: {type(value).__name__}")


def coerce_stage(stage: PhysicalDesignStage | str) -> PhysicalDesignStage:
    try:
        return stage if isinstance(stage, PhysicalDesignStage) else PhysicalDesignStage(stage)
    except ValueError as exc:
        raise ValueError(f"unknown physical-design stage: {stage!r}") from exc


class StageProgression:
    """Enforce one forward pass through the four cooperative stages."""

    def __init__(self) -> None:
        self._next_index = 0
        self.completed: list[PhysicalDesignStage] = []

    @property
    def next_stage(self) -> PhysicalDesignStage | None:
        if self._next_index == len(STAGE_ORDER):
            return None
        return STAGE_ORDER[self._next_index]

    @property
    def is_complete(self) -> bool:
        return self.next_stage is None

    def can_run(self, stage: PhysicalDesignStage | str) -> bool:
        return coerce_stage(stage) == self.next_stage

    def record_success(self, stage: PhysicalDesignStage | str) -> None:
        normalized = coerce_stage(stage)
        if not self.can_run(normalized):
            expected = self.next_stage.value if self.next_stage else "no stage"
            raise ValueError(f"expected {expected} (expected stage {expected}), received {normalized.value}")
        self.completed.append(normalized)
        self._next_index += 1

    def reset(self) -> None:
        self._next_index = 0
        self.completed.clear()


def normalize_stage_actions(
    actions: Mapping[PhysicalDesignStage | str, Sequence[Any]] | None,
    *,
    action_dim: int = 6,
) -> dict[PhysicalDesignStage, tuple[StageActionSpec | None, ...]]:
    """Normalize action catalogs; unspecified actions remain masked."""
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    normalized = {stage: [None] * action_dim for stage in STAGE_ORDER}
    for stage in STAGE_ORDER:
        normalized[stage][0] = StageActionSpec(name="baseline", requires_real_tool_verification=False)
    for raw_stage, values in (actions or {}).items():
        stage = coerce_stage(raw_stage)
        for index, value in enumerate(values):
            if index >= action_dim:
                raise ValueError(f"too many actions for {stage.value}; action_dim={action_dim}")
            normalized[stage][index] = StageActionSpec.from_value(value, index=index)
    return {stage: tuple(values) for stage, values in normalized.items()}
