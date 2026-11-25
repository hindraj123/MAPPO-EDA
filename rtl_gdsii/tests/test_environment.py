"""Synthetic environment integration tests only.

These tests exercise process orchestration and tensor-input shapes.  They are
not OpenROAD measurements; real-tool verification remains a Linux deployment
task.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import time

import numpy as np
import pytest

from mappo_eda.environment import (
    EnvironmentStatus,
    OpenROADCommandRunner,
    OpenROADEnvironment,
    PhysicalDesignStage,
)


def _synthetic_openroad(tmp_path: Path, *, sleep: bool = False) -> Path:
    executable = tmp_path / ("synthetic-timeout" if sleep else "synthetic-openroad")
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        + ("time.sleep(0.2)\n" if sleep else "")
        + "print('synthetic fixture: not a physical-design measurement')\n"
        + "print('stage=' + os.environ.get('MAPPO_EDA_STAGE', 'missing'))\n"
        + "print('action=' + os.environ.get('MAPPO_EDA_ACTION_INDEX', 'missing'))\n"
        + "print('MAPP0EDA_METRIC area_um2 10.0')\n"
        + "print('MAPP0EDA_METRIC wns_ns 1.0')\n"
        + "sys.exit(0)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _environment(tmp_path: Path, *, sleep: bool = False, stage_actions=None) -> OpenROADEnvironment:
    executable = _synthetic_openroad(tmp_path, sleep=sleep)
    scripts = {}
    for stage in PhysicalDesignStage:
        script = tmp_path / f"{stage.value}.tcl"
        script.write_text(f"# synthetic fixture for {stage.value}\n", encoding="utf-8")
        scripts[stage] = script
    return OpenROADEnvironment(
        design="synthetic-gcd",
        runner=OpenROADCommandRunner(str(executable)),
        workdir=tmp_path / "episodes",
        stage_scripts=scripts,
        action_dim=3,
        stage_actions=stage_actions,
        reward_scales={"area_um2": 10.0, "wns_ns": 1.0},
        reward_directions={"area_um2": -1.0, "wns_ns": 1.0},
    )


def test_synthetic_observation_is_stage_aware_and_masked(tmp_path: Path):
    env = _environment(tmp_path)
    observation = env.observation()
    assert observation.stage is PhysicalDesignStage.FLOORPLANNING
    assert observation.nodes.shape == (8, 4)
    assert observation.image.shape == (1, 8, 8)
    assert observation.scalars.shape == (8,)
    assert observation.global_state.shape == (16,)
    assert observation.action_mask.tolist() == [True, False, False]
    assert env.action_masks()["placement"] == [False, False, False]
    assert not any(observation.metric_presence.values())


def test_synthetic_stage_action_is_applied_to_openroad_process(tmp_path: Path):
    actions = {
        PhysicalDesignStage.FLOORPLANNING: [
            {"name": "baseline"},
            {"name": "parameterized", "environment": {"MAPPO_EDA_TEST_PARAMETER": "enabled"}, "post_commands": ["puts {synthetic action command}"]},
        ]
    }
    env = _environment(tmp_path, stage_actions=actions)
    result = env.run_stage(PhysicalDesignStage.FLOORPLANNING, action=1)
    assert result.succeeded
    assert "action=1" in result.result.stdout
    assert (env.workdir / "logs" / "openroad.stdout.log").is_file()


def test_synthetic_four_stage_transition_reaches_terminal(tmp_path: Path):
    env = _environment(tmp_path)
    workdirs = []
    for stage in PhysicalDesignStage:
        step = env.step(0)
        workdirs.append(step.stage_run.workdir)
        assert step.stage_run.succeeded
        assert step.info["stage"] == stage.value
    assert env.status is EnvironmentStatus.COMPLETE
    assert step.terminated
    assert len(set(workdirs)) == 1
    assert env.action_masks()["routing"] == [False, False, False]


def test_reset_allocates_an_isolated_episode_directory(tmp_path: Path):
    env = _environment(tmp_path)
    first = env.workdir
    env.reset()
    second = env.workdir
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.parent == second.parent == (tmp_path / "episodes").resolve()


def test_masked_action_and_tool_failure_are_explicit(tmp_path: Path):
    env = _environment(tmp_path)
    with pytest.raises(ValueError, match="masked"):
        env.step(1)
    missing = OpenROADEnvironment(
        design="synthetic-gcd",
        runner=OpenROADCommandRunner("missing-openroad-for-test"),
        workdir=tmp_path / "missing-episodes",
        stage_scripts={stage: tmp_path / "does-not-exist.tcl" for stage in PhysicalDesignStage},
    )
    status = missing.check_toolchain()
    assert not status.available
    assert missing.is_openroad_available() is False


def test_timeout_enters_failure_terminal_state(tmp_path: Path):
    env = _environment(tmp_path, sleep=True)
    result = env.run_stage(PhysicalDesignStage.FLOORPLANNING, timeout_s=0.01)
    assert not result.succeeded
    assert env.status is EnvironmentStatus.FAILED
    assert env.terminated
    assert "timed out" in (result.error or result.result.stderr)


def test_policy_inputs_are_numpy_and_metrics_are_only_emitted_values(tmp_path: Path):
    env = _environment(tmp_path)
    step = env.step(0)
    policy_inputs = step.observation.to_policy_inputs()
    assert all(isinstance(value, np.ndarray) for value in policy_inputs.values())
    assert step.observation.measured_metrics == {"area_um2": 10.0, "wns_ns": 1.0}
    assert "instance_count" not in step.observation.measured_metrics
