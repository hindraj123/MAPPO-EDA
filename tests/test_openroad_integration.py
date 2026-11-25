"""OpenROAD integration tests.

Tests named synthetic use a temporary fake executable and explicitly do not
represent physical-design measurements. Tests marked real require an actual
OpenROAD executable and skip on development machines where it is absent.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import sys

import pytest

from mappo_eda.environment import (
    OpenROADCommandRunner,
    OpenROADEnvironment,
    PhysicalDesignStage,
    StageProgression,
    build_reward_inputs,
)
from mappo_eda.extraction.metrics import MetricParseError, PhysicalDesignMetrics


def _synthetic_openroad(tmp_path: Path) -> Path:
    """Create a fake command executable for unit tests only."""

    executable = tmp_path / "synthetic-openroad"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('synthetic fixture: no physical-design measurement')\n"
        "print('MAPP0EDA_METRIC area_um2 10.0')\n"
        "print('MAPP0EDA_METRIC wns_ns 2.0')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def test_synthetic_runner_captures_output_and_logs(tmp_path: Path):
    executable = _synthetic_openroad(tmp_path)
    script = tmp_path / "stage.tcl"
    script.write_text("# synthetic fixture script\n", encoding="utf-8")

    result = OpenROADCommandRunner(str(executable)).run(
        script=script, cwd=tmp_path, log_dir=tmp_path / "logs"
    )

    assert result.succeeded
    assert "synthetic fixture" in result.stdout
    assert (tmp_path / "logs" / "openroad.stdout.log").read_text(encoding="utf-8") == result.stdout


def test_runner_sets_requested_thread_limit(tmp_path: Path):
    executable = _synthetic_openroad(tmp_path)
    result = OpenROADCommandRunner(str(executable), threads=2).run(commands=["puts synthetic"])
    assert result.succeeded
    assert result.command[1:5] == ("-no_init", "-exit", "-threads", "2")
    with pytest.raises(ValueError):
        OpenROADCommandRunner(str(executable), threads=0)


def test_metric_parser_preserves_only_emitted_values():
    metrics = PhysicalDesignMetrics.from_text(
        "noise\nMAPP0EDA_METRIC area_um2 12.5\nMAPP0EDA_METRIC instance_count 17\n"
    )
    assert metrics.to_dict() == {"area_um2": 12.5, "instance_count": 17.0}


def test_metric_parser_rejects_duplicate_records():
    with pytest.raises(MetricParseError, match="duplicate"):
        PhysicalDesignMetrics.from_text(
            "MAPP0EDA_METRIC area_um2 1\nMAPP0EDA_METRIC area_um2 2\n"
        )


def test_stage_progression_requires_four_ordered_stages():
    progression = StageProgression()
    assert progression.next_stage is PhysicalDesignStage.FLOORPLANNING
    with pytest.raises(ValueError, match="floorplanning"):
        progression.record_success(PhysicalDesignStage.ROUTING)
    for stage in (
        PhysicalDesignStage.FLOORPLANNING,
        PhysicalDesignStage.PLACEMENT,
        PhysicalDesignStage.CTS,
        PhysicalDesignStage.ROUTING,
    ):
        progression.record_success(stage)
    assert progression.is_complete


def test_reward_inputs_use_only_two_measured_snapshots():
    before = PhysicalDesignMetrics({"area_um2": 100.0, "wns_ns": -2.0})
    after = PhysicalDesignMetrics({"area_um2": 90.0, "wns_ns": -1.0})
    inputs = build_reward_inputs(
        PhysicalDesignStage.PLACEMENT,
        after,
        previous=before,
        scales={"area_um2": 10.0, "wns_ns": 1.0},
        directions={"area_um2": -1.0, "wns_ns": 1.0},
    )
    assert inputs.measured == {"area_um2": 90.0, "wns_ns": -1.0}
    assert inputs.deltas == {"area_um2": 10.0, "wns_ns": 1.0}
    assert inputs.normalized_deltas == {"area_um2": 1.0, "wns_ns": 1.0}


def test_synthetic_environment_runs_stages_and_extracts_metrics(tmp_path: Path):
    executable = _synthetic_openroad(tmp_path)
    stage_script = tmp_path / "stage.tcl"
    stage_script.write_text("# synthetic fixture script\n", encoding="utf-8")
    environment = OpenROADEnvironment(
        design="synthetic-gcd",
        runner=OpenROADCommandRunner(str(executable)),
        workdir=tmp_path,
        stage_scripts={stage: stage_script for stage in PhysicalDesignStage},
    )

    result = environment.run_stage(PhysicalDesignStage.FLOORPLANNING)

    assert result.succeeded
    assert result.metrics.to_dict() == {"area_um2": 10.0, "wns_ns": 2.0}
    assert environment.progression.next_stage is PhysicalDesignStage.PLACEMENT


@pytest.mark.skipif(shutil.which("openroad") is None, reason="real OpenROAD executable is not installed")
def test_real_openroad_executes_tcl_when_available(tmp_path: Path):
    executable = shutil.which("openroad")
    if executable is None:
        pytest.skip("real OpenROAD executable is not installed")
    script = tmp_path / "real_smoke.tcl"
    script.write_text("puts {MAPP0EDA_REAL_OPENROAD_SMOKE}\n", encoding="utf-8")
    result = OpenROADCommandRunner(executable).run(script=script, cwd=tmp_path)
    assert result.succeeded, result.stderr
    assert "MAPP0EDA_REAL_OPENROAD_SMOKE" in result.stdout


@pytest.mark.real_openroad
def test_real_gcd_database_metric_extraction(tmp_path: Path):
    executable = shutil.which("openroad")
    orfs_home = os.environ.get("ORFS_HOME")
    if not executable or not orfs_home:
        pytest.skip("requires OpenROAD and a completed ORFS GCD flow")
    flow = Path(orfs_home) / "flow"
    odb = flow / "results/nangate45/gcd/base/6_final.odb"
    if not odb.is_file():
        pytest.skip("completed GCD database is unavailable")
    extractor = Path(__file__).resolve().parents[1] / "scripts/openroad/extract_metrics.tcl"
    script = tmp_path / "extract.tcl"
    script.write_text('set_thread_count 2\nread_db $::env(MAPPO_TEST_ODB)\nsource $::env(MAPPO_TEST_EXTRACT)\n')
    result = OpenROADCommandRunner(executable).run(script=script, cwd=tmp_path, environment={"MAPPO_TEST_ODB": str(odb), "MAPPO_TEST_EXTRACT": str(extractor)})
    assert result.succeeded, result.stderr
    metrics = PhysicalDesignMetrics.from_text(result.stdout)
    assert metrics["die_width_um"] > 0 and metrics["die_height_um"] > 0
    assert metrics["instance_count"] > 0 and metrics["net_count"] > 0


@pytest.mark.skipif(shutil.which("openroad") is None, reason="real OpenROAD executable is not installed")
def test_gcd_flow_requires_real_deployment_inputs():
    if shutil.which("openroad") is None:
        pytest.skip("real OpenROAD executable is not installed")
    required = (
        "MAPPO_EDA_LEF",
        "MAPPO_EDA_LIBERTY",
        "MAPPO_EDA_GDS_MAP",
        "MAPPO_EDA_DIE_AREA",
        "MAPPO_EDA_CORE_AREA",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip("Linux technology deployment inputs are not configured")
    pytest.skip("full GCD execution is deployment-specific; run on Ubuntu with technology files")


def test_python_executable_available_for_synthetic_fixture():
    assert Path(sys.executable).is_file()
