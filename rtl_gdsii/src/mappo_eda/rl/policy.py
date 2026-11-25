"""Four decentralized stage actors and a centralized critic."""
from __future__ import annotations

from typing import Any
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from .encoders import CNNEncoder, GraphEncoder, MLPEncoder, MultiViewFusion


def _mask_logits(logits: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return logits
    mask = torch.broadcast_to(mask.to(logits.device, torch.bool), logits.shape)
    if not mask.any(dim=-1).all():
        raise ValueError("each batch item needs at least one valid action")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class StageActor(nn.Module):
    def __init__(self, node_dim: int, scalar_dim: int, image_channels: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        if action_dim <= 0: raise ValueError("action_dim must be positive")
        self.graph = GraphEncoder(node_dim, hidden_dim)
        self.image = CNNEncoder(image_channels, hidden_dim)
        self.scalar = MLPEncoder(scalar_dim, hidden_dim)
        self.fusion = MultiViewFusion(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)

    def encode(self, nodes, edges, image, scalars, node_mask=None):
        encoded = self.graph(nodes, edges, node_mask)
        if encoded.ndim == 2:
            graph = encoded.mean(0)
        elif node_mask is None:
            graph = encoded.mean(1)
        else:
            weights = torch.broadcast_to(node_mask.to(encoded.device, encoded.dtype), encoded.shape[:-1])
            graph = (encoded * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        return self.fusion(graph, self.image(image), self.scalar(scalars))

    def forward(self, nodes, edges, image, scalars, action_mask=None, node_mask=None):
        return _mask_logits(self.policy(self.encode(nodes, edges, image, scalars, node_mask)), action_mask)

    def distribution(self, nodes, edges, image, scalars, action_mask=None, node_mask=None):
        return Categorical(logits=self(nodes, edges, image, scalars, action_mask, node_mask))

    def act(self, nodes, edges, image, scalars, action_mask=None, node_mask=None, deterministic=False):
        dist = self.distribution(nodes, edges, image, scalars, action_mask, node_mask)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), dist.entropy()

    def evaluate_actions(self, nodes, edges, image, scalars, actions, action_mask=None, node_mask=None):
        dist = self.distribution(nodes, edges, image, scalars, action_mask, node_mask)
        return dist.log_prob(actions), dist.entropy()


class FloorplanningActor(StageActor): pass
class PlacementActor(StageActor): pass
class CTSActor(StageActor): pass
class RoutingActor(StageActor): pass


class CentralizedCritic(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.global_dim = global_dim
        self.value = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, global_state: Tensor) -> Tensor:
        if global_state.shape[-1] != self.global_dim: raise ValueError("wrong global state width")
        return self.value(global_state).squeeze(-1)


class MAPPOPolicy(nn.Module):
    STAGES = ("floorplanning", "placement", "cts", "routing")
    ACTOR_TYPES = {"floorplanning": FloorplanningActor, "placement": PlacementActor, "cts": CTSActor, "routing": RoutingActor}

    def __init__(self, actor_kwargs: dict[str, Any], global_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.actors = nn.ModuleDict({stage: self.ACTOR_TYPES[stage](**actor_kwargs) for stage in self.STAGES})
        self.critic = CentralizedCritic(global_dim, hidden_dim)

    def actor(self, stage: str) -> StageActor:
        if stage not in self.actors: raise KeyError(stage)
        return self.actors[stage]

    def value(self, global_state: Tensor) -> Tensor:
        return self.critic(global_state)
