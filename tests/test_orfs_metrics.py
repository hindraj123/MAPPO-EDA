"""Synthetic JSON verifies metric names; no fixture values are measurements."""
import json

from mappo_eda.extraction.orfs_metrics import read_orfs_metrics


def test_reads_only_selected_stage_and_distinguishes_estimates(tmp_path):
    path = tmp_path / "synthetic-metrics.json"
    path.write_text(json.dumps({"cts__timing__setup__ws": -0.2, "cts__route__wirelength__estimated": 3,
                                "detailedroute__route__wirelength": 4, "cts__power__total": None}))
    result = read_orfs_metrics(path, "cts").to_dict()
    assert result == {"worst_slack_ns": -0.2, "estimated_wirelength_um": 3}
