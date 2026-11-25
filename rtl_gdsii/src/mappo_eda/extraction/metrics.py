"""Strict parsing of machine-readable metrics emitted by OpenROAD Tcl."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping


class MetricParseError(ValueError):
    """Raised when an OpenROAD metric record is malformed or ambiguous."""


_PREFIX = "MAPP0EDA_METRIC "
_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PhysicalDesignMetrics:
    """Only metrics actually emitted by the tool are stored."""

    values: Mapping[str, float]

    def __post_init__(self) -> None:
        normalized = {str(name): float(value) for name, value in self.values.items()}
        for name, value in normalized.items():
            if not _KEY.fullmatch(name):
                raise MetricParseError(f"invalid metric name: {name!r}")
            if not math.isfinite(value):
                raise MetricParseError(f"metric {name!r} is not finite")
        object.__setattr__(self, "values", normalized)

    def __bool__(self) -> bool:
        return bool(self.values)

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)

    @classmethod
    def from_text(cls, text: str) -> "PhysicalDesignMetrics":
        parsed: dict[str, float] = {}
        for line in text.splitlines():
            if not line.startswith(_PREFIX):
                continue
            record = line[len(_PREFIX) :].strip().split()
            if len(record) != 2:
                raise MetricParseError(f"malformed metric record: {line!r}")
            name, raw_value = record
            if not _KEY.fullmatch(name):
                raise MetricParseError(f"invalid metric name: {name!r}")
            if name in parsed:
                raise MetricParseError(f"duplicate metric record: {name!r}")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise MetricParseError(f"metric {name!r} is not numeric") from exc
            if not math.isfinite(value):
                raise MetricParseError(f"metric {name!r} is not finite")
            parsed[name] = value
        return cls(parsed)
