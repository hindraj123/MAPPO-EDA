import json

from mappo_eda.extraction.schema import DesignState
from mappo_eda.logging import RunLogger


def test_design_state_is_serializable():
    state = DesignState(design="gcd", stage="floorplan", counts={"instances": 12})
    payload = json.loads(state.to_json())
    assert payload["design"] == "gcd"
    assert payload["counts"]["instances"] == 12


def test_logger_writes_jsonl(tmp_path):
    logger = RunLogger(tmp_path / "run")
    logger.event(kind="metric", wirelength=42.0)
    logger.failure(stage="route", error="timeout")
    assert json.loads((tmp_path / "run" / "events.jsonl").read_text())["wirelength"] == 42.0
    assert json.loads((tmp_path / "run" / "failures.jsonl").read_text())["stage"] == "route"
