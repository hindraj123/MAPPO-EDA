"""Read explicitly named metrics from an ORFS/OpenROAD stage report."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .metrics import PhysicalDesignMetrics


def read_orfs_metrics(path: Path | str, stage: str) -> PhysicalDesignMetrics:
    """Read one stage's JSON; omitted or non-finite values stay absent.

    Estimated and detailed wirelength deliberately use different names. Power
    retains the report's native units, which must be checked from its metadata.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not stage or "__" in stage:
        raise ValueError("expected an ORFS metrics object and a stage prefix")
    names = {
        "timing__setup__ws": "worst_slack_ns",
        "timing__setup__tns": "tns_ns",
        "design__instance__area": "area_um2",
        "power__total": "power_native",
        "route__wirelength__estimated": "estimated_wirelength_um",
        "route__wirelength": "wirelength_um",
        "route__drc_errors": "drc_violations",
    }
    result = {}
    for source, target in names.items():
        value = data.get(f"{stage}__{source}")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[target] = float(value)
    return PhysicalDesignMetrics(result)
