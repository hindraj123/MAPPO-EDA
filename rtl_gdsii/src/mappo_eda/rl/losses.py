"""Clipped PPO objectives used by MAPPO."""
from __future__ import annotations
import torch
from torch import Tensor

def masked_mean(value: Tensor, mask: Tensor | None = None) -> Tensor:
    if mask is None: return value.mean()
    weights = torch.broadcast_to(mask.to(value.device, value.dtype), value.shape)
    total = weights.sum()
    return value.sum() * 0 if total.item() == 0 else (value * weights).sum() / total

def ppo_loss(new_log_probs, old_log_probs, advantages, clip=.2, mask=None, normalize_advantages=True):
    if clip <= 0: raise ValueError("clip must be positive")
    if not (new_log_probs.shape == old_log_probs.shape == advantages.shape): raise ValueError("PPO tensors must have equal shapes")
    if normalize_advantages:
        if mask is None: mean, scale = advantages.mean(), advantages.std(unbiased=False).clamp_min(1e-8)
        else:
            w = torch.broadcast_to(mask.to(advantages.device, advantages.dtype), advantages.shape); n = w.sum().clamp_min(1)
            mean = (advantages*w).sum()/n; scale = (((advantages-mean).square()*w).sum()/n).sqrt().clamp_min(1e-8)
        advantages = (advantages-mean)/scale
    ratio = (new_log_probs-old_log_probs).exp()
    return -masked_mean(torch.minimum(ratio*advantages, ratio.clamp(1-clip, 1+clip)*advantages), mask)

def value_loss(values, returns, old_values=None, clip=None, mask=None):
    if values.shape != returns.shape: raise ValueError("values and returns must have equal shapes")
    if clip is None: loss = (values-returns).square()
    else:
        if old_values is None or old_values.shape != values.shape: raise ValueError("old_values required for clipping")
        clipped = old_values + (values-old_values).clamp(-clip, clip)
        loss = torch.maximum((values-returns).square(), (clipped-returns).square())
    return .5 * masked_mean(loss, mask)
