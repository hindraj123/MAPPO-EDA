from pathlib import Path
import pytest

from mappo_eda.execution.openroad import OpenROADNotFound, OpenROADRunner


def test_runner_rejects_empty_request():
    with pytest.raises(ValueError):
        OpenROADRunner("missing-openroad").run()


def test_runner_reports_missing_binary():
    with pytest.raises(OpenROADNotFound):
        OpenROADRunner("definitely-missing-openroad").run(commands=["puts hello"])


def test_script_and_commands_are_mutually_exclusive(tmp_path: Path):
    script = tmp_path / "x.tcl"
    script.write_text("puts hello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        OpenROADRunner("missing-openroad").run(script=script, commands=["puts hello"])
