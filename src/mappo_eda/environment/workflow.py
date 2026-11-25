"""OpenROAD environment and four-stage integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from mappo_eda.extraction.metrics import MetricParseError, PhysicalDesignMetrics
from mappo_eda.extraction.schema import DesignState

from .openroad import CommandResult, OpenROADCommandRunner, OpenROADNotFound
from .reward import RewardInputs, build_reward_inputs
from .stages import STAGE_ORDER, PhysicalDesignStage, StageActionSpec, StageProgression, coerce_stage, normalize_stage_actions


class EnvironmentStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolchainStatus:
    executable: str
    available: bool
    required_environment: tuple[str, ...] = ()
    missing_environment: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.available and not self.missing_environment


@dataclass(frozen=True)
class StageObservation:
    """CPU-friendly, fixed-shape GNN/CNN/MLP views for one stage agent."""

    stage: PhysicalDesignStage
    nodes: np.ndarray
    edges: np.ndarray
    image: np.ndarray
    scalars: np.ndarray
    global_state: np.ndarray
    node_mask: np.ndarray
    action_mask: np.ndarray
    metric_presence: dict[str, bool] = field(default_factory=dict)
    measured_metrics: dict[str, float] = field(default_factory=dict)

    def to_policy_inputs(self) -> dict[str, np.ndarray]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "image": self.image,
            "scalars": self.scalars,
            "global_state": self.global_state,
            "node_mask": self.node_mask,
            "action_mask": self.action_mask,
        }


@dataclass(frozen=True)
class StageRun:
    stage: PhysicalDesignStage
    result: CommandResult
    metrics: PhysicalDesignMetrics
    reward_inputs: RewardInputs | None
    state: DesignState
    action: int = 0
    error: str | None = None
    workdir: Path | None = None

    @property
    def succeeded(self) -> bool:
        return self.result.succeeded and self.error is None


@dataclass(frozen=True)
class EnvironmentStep:
    observation: StageObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    stage_run: StageRun

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield dict(self.info)


_METRIC_NAMES = ("area_um2", "die_width_um", "die_height_um", "instance_count", "net_count", "wns_ns", "tns_ns", "wirelength_um")
_METRIC_SCALES = {"area_um2": 1e6, "die_width_um": 1e3, "die_height_um": 1e3, "instance_count": 1e6, "net_count": 1e6, "wns_ns": 10.0, "tns_ns": 100.0, "wirelength_um": 1e6}


class OpenROADEnvironment:
    """Run floorplanning, placement, CTS and routing as one episode."""

    def __init__(
        self,
        *,
        design: str,
        runner: OpenROADCommandRunner,
        workdir: Path | str,
        stage_scripts: Mapping[PhysicalDesignStage | str, Path | str],
        extraction_script: Path | str | None = None,
        initialization_script: Path | str | None = None,
        state_db_name: str = "design.odb",
        reward_scales: Mapping[str, float] | None = None,
        reward_directions: Mapping[str, float] | None = None,
        stage_actions: Mapping[PhysicalDesignStage | str, Sequence[Any]] | None = None,
        action_dim: int = 6,
        required_environment: Sequence[str] = (),
        node_dim: int = 4,
        scalar_dim: int = 8,
        image_channels: int = 1,
        image_size: int = 8,
        keep_workdirs: bool = True,
    ) -> None:
        if node_dim < 4 or scalar_dim < 8 or image_channels != 1 or image_size <= 0:
            raise ValueError("observation dimensions must be node_dim>=4, scalar_dim>=8, image_channels=1")
        self.design = design
        self.runner = runner
        self.base_workdir = Path(workdir).resolve()
        self.base_workdir.mkdir(parents=True, exist_ok=True)
        self.workdir = self.base_workdir
        self.stage_scripts = {coerce_stage(k): Path(v).resolve() for k, v in stage_scripts.items()}
        self.extraction_script = Path(extraction_script).resolve() if extraction_script else None
        self.initialization_script = Path(initialization_script).resolve() if initialization_script else None
        self.state_db_name = state_db_name
        self.reward_scales = dict(reward_scales or {})
        self.reward_directions = dict(reward_directions or {})
        self.required_environment = tuple(str(item) for item in required_environment)
        self.action_dim = action_dim
        self.stage_actions = normalize_stage_actions(stage_actions, action_dim=action_dim)
        self.node_dim, self.scalar_dim, self.image_size = node_dim, scalar_dim, image_size
        self.image_channels = image_channels
        self.keep_workdirs = keep_workdirs
        self.progression = StageProgression()
        self.previous_metrics: PhysicalDesignMetrics | None = None
        self.status = EnvironmentStatus.READY
        self.failure_reason: str | None = None
        self.episode_id = 0
        self.episode_workdir: Path | None = None
        self.last_stage: PhysicalDesignStage | None = None
        self.last_state = DesignState(design=design, stage=STAGE_ORDER[0].value)
        self.reset()

    @property
    def terminated(self) -> bool:
        return self.status in {EnvironmentStatus.COMPLETE, EnvironmentStatus.FAILED}

    @property
    def next_stage(self) -> PhysicalDesignStage | None:
        return self.progression.next_stage

    @property
    def active_agent(self) -> str | None:
        stage = self.progression.next_stage
        return stage.value if stage is not None and not self.terminated else None

    def agent_observations(self) -> dict[str, StageObservation]:
        """Return stage-keyed observations for decentralized actors."""
        return {stage.value: self.observation(stage) for stage in STAGE_ORDER}

    def check_toolchain(self) -> ToolchainStatus:
        try:
            executable = self.runner.resolve()
            available, error = True, None
        except OpenROADNotFound as exc:
            executable, available, error = self.runner.executable, False, str(exc)
        missing = tuple(name for name in self.required_environment if not os.environ.get(name))
        return ToolchainStatus(executable, available, self.required_environment, missing, error)

    def is_openroad_available(self) -> bool:
        return self.check_toolchain().available

    def require_openroad(self) -> str:
        status = self.check_toolchain()
        if not status.available:
            raise OpenROADNotFound(status.error or "OpenROAD is unavailable")
        if status.missing_environment:
            raise RuntimeError("missing OpenROAD deployment inputs: " + ", ".join(status.missing_environment))
        return status.executable

    def reset(self, initial_metrics: PhysicalDesignMetrics | None = None) -> DesignState:
        self.progression.reset()
        self.previous_metrics = initial_metrics
        self.status, self.failure_reason, self.last_stage = EnvironmentStatus.READY, None, None
        self.episode_id += 1
        self.episode_workdir = Path(tempfile.mkdtemp(prefix=f"{_safe_name(self.design)}-", dir=self.base_workdir))
        self.workdir = self.episode_workdir
        metrics = initial_metrics.to_dict() if initial_metrics is not None else {}
        self.last_state = DesignState(design=self.design, stage=STAGE_ORDER[0].value, metrics=metrics, action_mask=self.action_masks(), metadata={"episode_id": self.episode_id, "synthetic_fixture": False})
        return self.last_state

    def action_masks(self) -> dict[str, list[bool]]:
        masks = {stage.value: [False] * self.action_dim for stage in STAGE_ORDER}
        if self.terminated:
            return masks
        stage = self.progression.next_stage
        if stage is not None:
            masks[stage.value] = [spec is not None for spec in self.stage_actions[stage]]
        return masks

    def observation(self, stage: PhysicalDesignStage | str | None = None) -> StageObservation:
        selected = coerce_stage(stage) if stage is not None else (self.progression.next_stage or self.last_stage or STAGE_ORDER[-1])
        metrics = self.last_state.metrics
        presence = {name: name in metrics for name in _METRIC_NAMES}
        nodes = np.zeros((len(_METRIC_NAMES), self.node_dim), dtype=np.float32)
        node_mask = np.asarray([presence[name] for name in _METRIC_NAMES], dtype=np.float32)
        for index, name in enumerate(_METRIC_NAMES):
            if presence[name]:
                nodes[index, 0] = float(metrics[name]) / _METRIC_SCALES[name]
            nodes[index, 1] = node_mask[index]
            nodes[index, 2] = float(STAGE_ORDER.index(selected)) / (len(STAGE_ORDER) - 1)
            nodes[index, 3] = float(self.status == EnvironmentStatus.FAILED)
        edges = np.asarray([range(len(_METRIC_NAMES)), [(i + 1) % len(_METRIC_NAMES) for i in range(len(_METRIC_NAMES))]], dtype=np.int64)
        image = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        for index, name in enumerate(_METRIC_NAMES):
            if presence[name]:
                row, col = divmod(index, self.image_size)
                image[0, row, col] = nodes[index, 0]
        scalars = np.zeros(self.scalar_dim, dtype=np.float32)
        scalars[STAGE_ORDER.index(selected)] = 1.0
        scalars[4:8] = (STAGE_ORDER.index(selected) / 3.0, float(self.terminated), float(self.status == EnvironmentStatus.FAILED), sum(presence.values()) / len(_METRIC_NAMES))
        global_state = np.zeros(max(16, self.scalar_dim * 2), dtype=np.float32)
        global_state[: self.scalar_dim] = scalars
        for index, name in enumerate(_METRIC_NAMES):
            if presence[name]:
                global_state[self.scalar_dim + index] = nodes[index, 0]
        return StageObservation(selected, nodes, edges, image, scalars, global_state, node_mask, np.asarray(self.action_masks()[selected.value], dtype=bool), presence, dict(metrics))

    def get_observation(self, stage: PhysicalDesignStage | str | None = None) -> StageObservation:
        return self.observation(stage)

    def run_stage(self, stage: PhysicalDesignStage | str, *, action: int = 0, stage_script: Path | str | None = None, extraction_script: Path | str | None = None, timeout_s: float = 1800.0, environment: Mapping[str, str] | None = None, log_dir: Path | str | None = None) -> StageRun:
        normalized = coerce_stage(stage)
        if not self.progression.can_run(normalized):
            expected = self.progression.next_stage.value if self.progression.next_stage else "none"
            raise ValueError(f"stage order violation: expected {expected}, got {normalized.value}")
        spec = self._action_spec(normalized, action)
        selected_stage = Path(stage_script).resolve() if stage_script else self.stage_scripts.get(normalized)
        selected_extract = Path(extraction_script).resolve() if extraction_script else self.extraction_script
        if selected_stage is None or not selected_stage.is_file():
            return self._failure(normalized, action, f"stage script is unavailable: {selected_stage}")
        if selected_extract is not None and not selected_extract.is_file():
            return self._failure(normalized, action, f"extraction script is unavailable: {selected_extract}")
        if self.initialization_script is not None and not self.initialization_script.is_file():
            return self._failure(normalized, action, f"initialization script is unavailable: {self.initialization_script}")
        assert self.episode_workdir is not None
        env = {str(k): str(v) for k, v in (environment or {}).items()}
        env.update(spec.environment)
        env.update({"MAPPO_EDA_DESIGN": self.design, "MAPPO_EDA_STAGE": normalized.value, "MAPPO_EDA_ACTION_INDEX": str(action), "MAPPO_EDA_WORKDIR": str(self.episode_workdir)})
        output_dir = Path(log_dir) if log_dir else self.episode_workdir / "logs"
        try:
            combined = self._compose_scripts(selected_stage, selected_extract, spec)
        except OSError as exc:
            return self._failure(normalized, action, str(exc))
        self.status = EnvironmentStatus.RUNNING
        try:
            result = self.runner.run(script=combined, cwd=self.episode_workdir, timeout_s=timeout_s, environment=env, log_dir=output_dir)
        except (OpenROADNotFound, OSError) as exc:
            return self._failure(normalized, action, str(exc))
        finally:
            combined.unlink(missing_ok=True)
        try:
            metrics = PhysicalDesignMetrics.from_text(result.stdout)
        except MetricParseError as exc:
            return self._failure(normalized, action, f"metric extraction failed: {exc}", result=result)
        if not result.succeeded:
            return self._failure(normalized, action, result.stderr.strip() or f"OpenROAD exited with {result.returncode}", result=result, metrics=metrics)
        reward_inputs = build_reward_inputs(normalized, metrics, self.previous_metrics, self.reward_scales, self.reward_directions) if metrics else None
        self.progression.record_success(normalized)
        self.previous_metrics = metrics if metrics else self.previous_metrics
        self.last_stage = normalized
        self.status = EnvironmentStatus.COMPLETE if self.progression.is_complete else EnvironmentStatus.READY
        self.last_state = DesignState(design=self.design, stage=normalized.value, metrics=metrics.to_dict(), action_mask=self.action_masks(), metadata={"episode_id": self.episode_id, "command": list(result.command), "duration_s": result.duration_s, "measured": bool(metrics)})
        return StageRun(normalized, result, metrics, reward_inputs, self.last_state, action, None, self.episode_workdir)

    def step(self, action: int, *, timeout_s: float = 1800.0, environment: Mapping[str, str] | None = None) -> EnvironmentStep:
        stage = self.progression.next_stage
        if stage is None or self.terminated:
            raise RuntimeError("cannot step a terminated environment; call reset()")
        run = self.run_stage(stage, action=action, timeout_s=timeout_s, environment=environment)
        reward = float(sum(run.reward_inputs.normalized_deltas.values())) if run.reward_inputs else 0.0
        info = {"stage": stage.value, "action": action, "metrics": run.metrics.to_dict(), "succeeded": run.succeeded, "error": run.error, "workdir": str(self.workdir), "real_tool_verified": False}
        return EnvironmentStep(self.observation(), reward, self.terminated, False, info, run)

    def _action_spec(self, stage: PhysicalDesignStage, action: int) -> StageActionSpec:
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_dim:
            raise ValueError(f"action must be an integer in [0, {self.action_dim})")
        spec = self.stage_actions[stage][int(action)]
        if spec is None:
            raise ValueError(f"action {action} is masked for {stage.value}")
        return spec

    def _compose_scripts(self, stage: Path, extraction: Path | None, spec: StageActionSpec) -> Path:
        assert self.episode_workdir is not None and self.progression.next_stage is not None
        path = self.episode_workdir / f"stage-{self.progression.next_stage.value}.tcl"
        lines = list(spec.pre_commands)
        if self.initialization_script is not None:
            if not self.initialization_script.is_file():
                raise FileNotFoundError(self.initialization_script)
            if self.progression.next_stage is STAGE_ORDER[0]:
                lines.append(f"source [file normalize {_tcl_word(self.initialization_script)}]")
            else:
                previous_db = self.episode_workdir / self.state_db_name
                lines.append(f"read_db {_tcl_word(previous_db)}")
        lines.append(f"source [file normalize {_tcl_word(stage)}]")
        lines.extend(spec.post_commands)
        if self.initialization_script is not None:
            lines.append(f"write_db {_tcl_word(self.episode_workdir / self.state_db_name)}")
        if extraction is not None:
            lines.append(f"source [file normalize {_tcl_word(extraction)}]")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _failure(self, stage: PhysicalDesignStage, action: int, error: str, *, result: CommandResult | None = None, metrics: PhysicalDesignMetrics | None = None) -> StageRun:
        result = result or CommandResult((self.runner.executable,), 127, "", error, 0.0)
        self.status, self.failure_reason = EnvironmentStatus.FAILED, error
        measured = metrics or PhysicalDesignMetrics({})
        self.last_state = DesignState(design=self.design, stage=stage.value, metrics=measured.to_dict(), action_mask=self.action_masks(), metadata={"episode_id": self.episode_id, "failure": error, "measured": bool(measured)})
        return StageRun(stage, result, measured, None, self.last_state, action, error, self.episode_workdir)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "design"


def _tcl_word(path: Path) -> str:
    value = str(path).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("[", "\\[").replace("]", "\\]")
    return f'"{value}"'
