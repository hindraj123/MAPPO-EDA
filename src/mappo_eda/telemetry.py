"""Parent-process experiment telemetry; unavailable measurements are omitted."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch


def public_config(value):
    """Remove credentials from configuration snapshots and checkpoints."""
    if isinstance(value, dict):
        blocked = ("api_key", "apikey", "password", "secret", "token", "credential")
        return {key: public_config(item) for key, item in value.items() if not any(word in str(key).lower() for word in blocked)}
    if isinstance(value, list):
        return [public_config(item) for item in value]
    return value


def start_wandb(config, run_dir, run_id=None):
    """Start or resume an online run using the SDK's local authentication."""
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B is enabled but the wandb package is unavailable in the existing Python environment") from exc
    if config.get("resume_checkpoint") and not run_id:
        raise ValueError("checkpoint has no W&B run ID; cannot resume the original online run")
    # Do not forward arbitrary configuration: it can contain credentials.
    safe_config = {key: config[key] for key in ("seed", "device", "rollout_workers", "rollout_steps", "learning_rate", "ppo_clip", "gamma", "gae_lambda") if key in config}
    args = dict(project="MAPPO-EDA", name="gcd-nangate45-mappo-4env", group="real-openroad-training", mode="online", tags=["MAPPO", "OpenROAD", "GCD", "Nangate45", "RTX4090"], config=safe_config, dir=str(run_dir))
    if run_id:
        args.update(id=run_id, resume="must")
    return wandb.init(**args)


def system_metrics(run_dir, device):
    result = {}
    usage = shutil.disk_usage(Path(run_dir))
    result["system/disk_utilization_percent"] = 100 * usage.used / usage.total
    try:
        import psutil
    except ImportError:
        pass
    else:
        result["system/cpu_utilization_percent"] = psutil.cpu_percent(interval=None)
        result["system/ram_utilization_percent"] = psutil.virtual_memory().percent
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        result["system/gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
        result["system/gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
        # NVML is optional; UUID matching avoids querying the wrong GPU when
        # CUDA_VISIBLE_DEVICES remaps the logical index.
        try:
            import pynvml
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            if uuid:
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid).encode())
                    result["system/gpu_utilization_percent"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    result["system/gpu_memory_used_bytes"] = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                finally:
                    pynvml.nvmlShutdown()
        except Exception:
            pass
    return result


def physical_metrics(measured, baseline=None, directions=None):
    """Expose measured values and only defined baseline-relative comparisons."""
    result = {}
    baseline, directions = baseline or {}, directions or {}
    for key, value in measured.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[f"physical/{key}"] = float(value)
            ref = baseline.get(key)
            direction = directions.get(key)
            if isinstance(ref, (int, float)) and math.isfinite(ref) and ref != 0 and direction in (-1, 1):
                result[f"improvement/{key}_percent"] = 100 * direction * (value - ref) / abs(ref)
    return result
