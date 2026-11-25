"""Measured metric deltas supplied to the shared reward function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mappo_eda.extraction.metrics import PhysicalDesignMetrics
from mappo_eda.environment.stages import PhysicalDesignStage, coerce_stage


@dataclass(frozen=True)
class RewardInputs:
    """Observed metrics and optional, explicitly normalized improvements.

    No value is imputed.  A metric appears only when it was emitted by
    OpenROAD, and a normalized delta appears only when both observations and a
    positive scale were supplied by the caller.
    """

    stage: PhysicalDesignStage
    measured: dict[str, float]
    deltas: dict[str, float]
    normalized_deltas: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "measured": dict(self.measured),
            "deltas": dict(self.deltas),
            "normalized_deltas": dict(self.normalized_deltas),
        }


def build_reward_inputs(
    stage: PhysicalDesignStage | str,
    current: PhysicalDesignMetrics,
    previous: PhysicalDesignMetrics | None = None,
    scales: Mapping[str, float] | None = None,
    directions: Mapping[str, float] | None = None,
) -> RewardInputs:
    """Build reward inputs from two measured OpenROAD metric snapshots.

    ``directions`` uses ``+1`` for higher-is-better and ``-1`` for
    lower-is-better.  It is applied only to deltas; it does not manufacture a
    reward when a metric is absent.  Scales are provided by the experiment
    configuration and are not inferred from a single observation.
    """

    normalized_stage = coerce_stage(stage)
    current_values = current.to_dict()
    previous_values = previous.to_dict() if previous is not None else {}
    deltas: dict[str, float] = {}
    for name, value in current_values.items():
        if name in previous_values:
            direction = 1.0 if directions is None else float(directions.get(name, 1.0))
            if direction not in (-1.0, 1.0):
                raise ValueError(f"reward direction for {name!r} must be -1 or 1")
            deltas[name] = direction * (value - previous_values[name])

    normalized: dict[str, float] = {}
    if scales is not None:
        for name, delta in deltas.items():
            if name not in scales:
                continue
            scale = float(scales[name])
            if scale <= 0:
                raise ValueError(f"reward scale for {name!r} must be positive")
            normalized[name] = delta / scale

    return RewardInputs(normalized_stage, current_values, deltas, normalized)
