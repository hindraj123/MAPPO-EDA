"""Portable, per-run MAPPO checkpoint helpers.

Checkpoints contain the complete trainable policy (all four actors, their
GNN/CNN/MLP encoders and the centralized critic), optimizer state, progress,
configuration, and RNG states.  They are ordinary per-run files; this module
does not create a trajectory store or an offline dataset.
"""

from __future__ import annotations

from pathlib import Path
import random
import shutil
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore RNG state saved by :func:`save_checkpoint`, when present."""

    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path | str,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    progress: Mapping[str, Any],
    seed: int,
    metadata: Mapping[str, Any] | None = None,
    scheduler: Any | None = None,
    policy_version: int | None = None,
    best_metric: Mapping[str, Any] | float | int | None = None,
) -> Path:
    """Atomically save a complete MAPPO training checkpoint.

    ``policy.state_dict()`` is the authoritative representation and includes
    every actor, each actor's GNN/CNN/MLP/fusion modules, and the centralized
    critic.  The explicit component snapshots make the artifact inspectable
    without changing the loading contract.  ``best_metric`` is metadata from a
    valid evaluation; no metric is computed or inferred here.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    policy_state = policy.state_dict()
    optimizer_state = optimizer.state_dict()
    actors = {
        name: policy.actor(name).state_dict()
        for name in getattr(policy, "STAGES", ())
        if hasattr(policy, "actor")
    }
    critic = policy.critic.state_dict() if hasattr(policy, "critic") else None
    encoders = {}
    for name in actors:
        actor = policy.actor(name)
        encoders[name] = {
            component: getattr(actor, component).state_dict()
            for component in ("graph", "image", "scalar", "fusion")
            if hasattr(actor, component)
        }
    resolved_policy_version = int(
        policy_version
        if policy_version is not None
        else progress.get("policy_version", 0)
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        # Keep both names for compatibility with early project artifacts.
        "model": policy_state,
        "policy_state_dict": policy_state,
        "actors": actors,
        "critic": critic,
        "encoders": encoders,
        "optimizer": optimizer_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": dict(config),
        "progress": dict(progress),
        "training_progress": dict(progress),
        "policy_version": resolved_policy_version,
        "best_metric": best_metric,
        "best_metrics": best_metric,
        "seed": int(seed),
        "metadata": dict(metadata or {}),
        "rng_state": _rng_state(),
    }
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(
    path: Path | str,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load policy, optimizer and progress state into live objects.

    The returned payload is retained for callers that need metadata.  Missing
    optimizer state is rejected when an optimizer is supplied so a partial
    checkpoint cannot silently resume a different optimization process.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    version = int(payload.get("format_version", 0))
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {version} is newer than supported format "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )
    state = payload.get("policy_state_dict", payload.get("model"))
    if state is None:
        raise KeyError("checkpoint does not contain policy/model state")
    policy.load_state_dict(state)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state_dict", payload.get("optimizer"))
        if optimizer_state is None:
            raise KeyError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state(optimizer, map_location)
    if scheduler is not None:
        scheduler_state = payload.get("scheduler_state_dict", payload.get("scheduler"))
        if scheduler_state is None:
            raise KeyError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))
    return payload


def checkpoint_progress(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized progress, including the explicit policy version."""

    progress = dict(payload.get("progress", payload.get("training_progress", {})))
    if "policy_version" not in progress and "policy_version" in payload:
        progress["policy_version"] = int(payload["policy_version"])
    if "best_metric" not in progress and "best_metric" in payload:
        progress["best_metric"] = payload["best_metric"]
    return progress


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    target = torch.device(device)
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(target)


def preserve_best_gdsii(source: Path | str, destination: Path | str) -> Path:
    """Copy a valid generated GDSII artifact to a per-run best-layout path."""

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"cannot preserve missing GDSII artifact: {source_path}")
    if source_path.suffix.lower() != ".gds":
        raise ValueError(f"expected a GDSII artifact with .gds suffix: {source_path}")
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "checkpoint_progress",
    "load_checkpoint",
    "preserve_best_gdsii",
    "restore_rng_state",
    "save_checkpoint",
]
