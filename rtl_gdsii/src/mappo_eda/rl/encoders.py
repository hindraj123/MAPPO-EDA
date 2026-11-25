"""Batched, dependency-light encoders for MAPPO state views.

The encoders accept arbitrary leading batch dimensions wherever that is
meaningful.  The graph encoder accepts either ``[nodes, features]`` or
``[batch, nodes, features]`` with one shared edge list.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MLPEncoder(nn.Module):
    """Encode scalar/vector features while preserving leading dimensions."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dimension {self.input_dim}, got {x.shape[-1]}")
        return self.net(x)


class CNNEncoder(nn.Module):
    """Encode a spatial view with ``[C,H,W]`` or ``[B,C,H,W]`` input."""

    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        if channels <= 0 or hidden_dim <= 0:
            raise ValueError("channels and hidden_dim must be positive")
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim not in (3, 4):
            raise ValueError("CNN input must have shape [C,H,W] or [B,C,H,W]")
        if x.shape[-3] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[-3]}")
        encoded = self.net(x.unsqueeze(0) if x.ndim == 3 else x)
        return encoded[0] if x.ndim == 3 else encoded


class GraphEncoder(nn.Module):
    """One mean-aggregation message-passing layer.

    ``edge_index`` follows the PyG convention ``[2, edges]`` and is shared by
    every item in a batch. ``node_mask`` masks padded nodes after encoding.
    """

    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        if node_dim <= 0 or hidden_dim <= 0:
            raise ValueError("node_dim and hidden_dim must be positive")
        self.node_dim = node_dim
        self.node = nn.Linear(node_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)

    @staticmethod
    def _validate_edges(edge_index: Tensor, nodes: Tensor) -> Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        edge_index = edge_index.to(device=nodes.device, dtype=torch.long)
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= nodes.shape[-2]):
            raise ValueError("edge_index contains a node outside the node feature tensor")
        return edge_index

    def forward(self, nodes: Tensor, edge_index: Tensor, node_mask: Tensor | None = None) -> Tensor:
        if nodes.ndim not in (2, 3):
            raise ValueError("nodes must have shape [N,F] or [B,N,F]")
        if nodes.shape[-1] != self.node_dim:
            raise ValueError(f"expected node feature dimension {self.node_dim}, got {nodes.shape[-1]}")
        edge_index = self._validate_edges(edge_index, nodes)
        source, destination = edge_index
        h = F.relu(self.node(nodes))
        messages = torch.zeros_like(h)
        degree = torch.zeros(nodes.shape[-2], device=nodes.device, dtype=h.dtype)
        if source.numel():
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=h.dtype))
            if nodes.ndim == 2:
                messages.index_add_(0, destination, h.index_select(0, source))
            else:
                messages.index_add_(1, destination, h.index_select(1, source))
            view_shape = (nodes.shape[-2], 1) if nodes.ndim == 2 else (1, nodes.shape[-2], 1)
            messages = messages / degree.clamp_min(1).view(view_shape)
        encoded = F.relu(self.update(torch.cat((h, messages), dim=-1)))
        if node_mask is not None:
            mask = node_mask.to(device=encoded.device, dtype=encoded.dtype)
            try:
                mask = torch.broadcast_to(mask, encoded.shape[:-1])
            except RuntimeError as exc:
                raise ValueError("node_mask is not broadcastable to the node batch shape") from exc
            encoded = encoded * mask.unsqueeze(-1)
        return encoded


class MultiViewFusion(nn.Module):
    """Fuse graph, spatial and scalar embeddings of equal width."""

    def __init__(self, view_dim: int, fused_dim: int):
        super().__init__()
        if view_dim <= 0 or fused_dim <= 0:
            raise ValueError("view_dim and fused_dim must be positive")
        self.view_dim = view_dim
        self.proj = nn.Sequential(nn.Linear(view_dim * 3, fused_dim), nn.LayerNorm(fused_dim), nn.Tanh())

    def forward(self, graph: Tensor, image: Tensor, scalar: Tensor) -> Tensor:
        for name, value in (("graph", graph), ("image", image), ("scalar", scalar)):
            if value.shape[-1] != self.view_dim:
                raise ValueError(f"{name} embedding has width {value.shape[-1]}, expected {self.view_dim}")
        try:
            graph, image, scalar = torch.broadcast_tensors(graph, image, scalar)
        except RuntimeError as exc:
            raise ValueError("multi-view embeddings have incompatible batch shapes") from exc
        return self.proj(torch.cat((graph, image, scalar), dim=-1))
