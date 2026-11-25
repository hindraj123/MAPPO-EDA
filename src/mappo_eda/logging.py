"""Lightweight per-run experiment logging; no persistent trajectory store."""

from pathlib import Path
import json
from typing import Any


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = run_dir / "events.jsonl"
        self.failures_path = run_dir / "failures.jsonl"

    def event(self, **data: Any) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, sort_keys=True) + "\n")

    def failure(self, **data: Any) -> None:
        with self.failures_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, sort_keys=True) + "\n")
