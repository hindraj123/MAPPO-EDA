"""Strictly on-policy rollout storage and generalized advantage estimation."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class RolloutBatch:
    observations: Tensor; actions: Tensor; log_probs: Tensor; rewards: Tensor
    values: Tensor; returns: Tensor; advantages: Tensor; masks: Tensor; dones: Tensor
    def to(self, device): return RolloutBatch(*(x.to(device) for x in self.__dict__.values()))

class OnPolicyRolloutBuffer:
    REQUIRED = ("observations", "actions", "log_probs", "rewards", "values")
    def __init__(self, horizon, workers, agents, device="cpu"):
        if min(horizon, workers, agents) <= 0: raise ValueError("sizes must be positive")
        self.horizon, self.workers, self.agents, self.device = horizon, workers, agents, torch.device(device)
        self.clear()
    def clear(self): self.items = []
    def __len__(self): return len(self.items)
    def add(self, **item):
        if len(self.items) >= self.horizon: raise RuntimeError("rollout horizon is full")
        missing = [k for k in self.REQUIRED if k not in item]
        if missing: raise KeyError(f"missing rollout fields: {missing}")
        self.items.append({k: torch.as_tensor(v).detach().to(self.device) for k, v in item.items()})
    @staticmethod
    def _expand(value, target):
        while value.ndim < target.ndim: value = value.unsqueeze(-1)
        try: return torch.broadcast_to(value, target.shape)
        except RuntimeError as e: raise ValueError(f"cannot broadcast {value.shape} to {target.shape}") from e
    def compute_gae(self, last_value, gamma=.99, gae_lambda=.95):
        if not self.items: raise RuntimeError("cannot compute GAE on an empty rollout")
        rewards = torch.stack([x["rewards"] for x in self.items])
        values = self._expand(torch.stack([x["values"] for x in self.items]), rewards)
        dones = self._expand(torch.stack([x.get("dones", torch.zeros_like(x["rewards"])) for x in self.items]), rewards).to(rewards.dtype)
        last = self._expand(torch.as_tensor(last_value, device=rewards.device, dtype=rewards.dtype), rewards[0])
        advantages, gae = torch.zeros_like(rewards), torch.zeros_like(last)
        for t in reversed(range(len(self.items))):
            not_done = 1 - dones[t]
            next_value = last if t == len(self.items)-1 else values[t+1]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
        return RolloutBatch(torch.stack([x["observations"] for x in self.items]), torch.stack([x["actions"] for x in self.items]), torch.stack([x["log_probs"] for x in self.items]), rewards, values, advantages + values, advantages, torch.stack([x.get("masks", torch.ones_like(x["rewards"])) for x in self.items]), dones)
