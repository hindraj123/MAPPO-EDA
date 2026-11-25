"""OpenROAD-backed environment primitives.

This package is deliberately independent from the policy and training code.  It
only turns an OpenROAD invocation into measured state and stage transitions.
"""

from .openroad import (
    CommandResult,
    OpenROADCommandRunner,
    OpenROADExecutionError,
    OpenROADNotFound,
)
from .reward import RewardInputs, build_reward_inputs
from .stages import PhysicalDesignStage, StageActionSpec, StageProgression, STAGE_AGENTS, STAGE_ORDER
from .workflow import (
    EnvironmentStatus,
    EnvironmentStep,
    OpenROADEnvironment,
    StageObservation,
    StageRun,
    ToolchainStatus,
)

__all__ = [
    "CommandResult",
    "OpenROADCommandRunner",
    "OpenROADExecutionError",
    "OpenROADNotFound",
    "OpenROADEnvironment",
    "EnvironmentStatus",
    "EnvironmentStep",
    "PhysicalDesignStage",
    "StageActionSpec",
    "StageObservation",
    "RewardInputs",
    "StageProgression",
    "StageRun",
    "ToolchainStatus",
    "STAGE_AGENTS",
    "STAGE_ORDER",
    "build_reward_inputs",
]
