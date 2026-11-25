"""Capture measured GCD baseline metrics from a completed ORFS run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mappo_eda.extraction.orfs_metrics import read_orfs_metrics


def capture(orfs_home: Path) -> dict:
    flow = orfs_home / "flow"
    root = flow / "logs/nangate45/gcd/base"
    finish = root / "6_report.json"
    route = root / "5_2_route.json"
    gds = flow / "results/nangate45/gcd/base/6_final.gds"
    if not gds.is_file() or gds.stat().st_size == 0:
        raise FileNotFoundError("a completed, nonempty GCD GDSII is required")
    metrics = read_orfs_metrics(finish, "finish").to_dict()
    metrics.update(read_orfs_metrics(route, "detailedroute").to_dict())
    if not metrics:
        raise ValueError("the ORFS reports did not contain recognized measurements")
    report = json.loads(finish.read_text(encoding="utf-8"))
    return {"design": "gcd", "platform": "nangate45", "synthetic_fixture": False,
            "metrics": metrics, "power_unit": report.get("run__flow__platform__power_units"),
            "gds_sha256": hashlib.sha256(gds.read_bytes()).hexdigest(),
            "sources": [str(finish.resolve()), str(route.resolve())]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orfs-home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = capture(args.orfs_home)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
