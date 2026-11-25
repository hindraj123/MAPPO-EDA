"""Deterministic synthetic evaluation utilities for development only."""
from __future__ import annotations
from typing import Any
import torch
from .rl.policy import MAPPOPolicy

def build_policy(config: dict[str, Any], device: str | torch.device = "cpu") -> MAPPOPolicy:
    policy = MAPPOPolicy({k: int(config[k]) for k in ("node_dim", "scalar_dim", "image_channels", "action_dim", "hidden_dim")}, global_dim=int(config["global_dim"]), hidden_dim=int(config["hidden_dim"])).to(device)
    return policy

def deterministic_rollout(policy: MAPPOPolicy, config: dict[str, Any], seed: int, baseline: str | None = None) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed); device = next(policy.parameters()).device; tensor_generator = torch.Generator(device=device).manual_seed(seed); total = 0.0; actions = []
    for index, stage in enumerate(MAPPOPolicy.STAGES):
        nodes = torch.randn(1, 5, int(config["node_dim"]), generator=tensor_generator, device=device); image = torch.randn(1, int(config["image_channels"]), 8, 8, generator=tensor_generator, device=device); scalars = torch.randn(1, int(config["scalar_dim"]), generator=tensor_generator, device=device); edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
        mask = torch.zeros(1, int(config["action_dim"]), dtype=torch.bool, device=device); mask[:, :max(1, int(config["action_dim"])-index)] = True
        if baseline == "first_valid": action = torch.zeros(1, dtype=torch.long, device=device)
        elif baseline == "last_valid": action = mask.sum(1).long() - 1
        elif baseline == "random": action = torch.multinomial(mask.float().cpu(), 1, generator=generator).to(device).reshape(-1)
        else: action, _, _ = policy.actor(stage).act(nodes, edges, image, scalars, mask, deterministic=True)
        actions.append(int(action.item())); total += 1.0 - abs(float(action.item()) - ((index + 1) % int(config["action_dim"]))) / max(1, int(config["action_dim"]))
    return {"return": total, "length": 4, "actions": actions, "synthetic_fixture": True}

def evaluate_synthetic(policy: MAPPOPolicy, config: dict[str, Any], episodes: int = 4, seed: int = 7, baselines: bool = True) -> dict[str, Any]:
    policy.eval(); results = {"policy": [deterministic_rollout(policy, config, seed + i) for i in range(episodes)]}
    if baselines:
        for name in ("first_valid", "last_valid", "random"): results[name] = [deterministic_rollout(policy, config, seed + i, name) for i in range(episodes)]
    return {"mode": "synthetic", "synthetic_fixture": True, "episodes": results, "mean_return": {k: sum(x["return"] for x in v) / len(v) for k, v in results.items()}}

def restore_for_evaluation(checkpoint, device="cpu"):
    payload = torch.load(checkpoint, map_location=device, weights_only=False); config = payload["config"]; policy = build_policy(config, device); policy.load_state_dict(payload.get("policy_state_dict", payload.get("model"))); policy.eval(); return policy, config, payload
