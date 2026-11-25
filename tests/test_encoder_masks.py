"""Synthetic feature-mask and batching checks; these are not EDA measurements."""
import pytest
import torch

from mappo_eda.rl.encoders import GraphEncoder
from mappo_eda.rl.policy import StageActor


@pytest.mark.parametrize("batched", [False, True])
def test_masked_nodes_cannot_change_valid_node_embeddings(batched):
    torch.manual_seed(4)
    encoder = GraphEncoder(2, 4)
    nodes = torch.randn(3, 2)
    changed = nodes.clone()
    changed[2] = 1000
    edges = torch.tensor([[0, 2], [1, 1]])
    mask = torch.tensor([True, True, False])
    if batched:
        nodes, changed, mask = nodes[None], changed[None], mask[None]
    assert torch.allclose(encoder(nodes, edges, mask), encoder(changed, edges, mask))


def test_masked_unbatched_graph_pooling_ignores_padding():
    torch.manual_seed(2)
    actor = StageActor(2, 2, 1, 3, 8)
    nodes = torch.randn(2, 2)
    image, scalar = torch.randn(1, 4, 4), torch.randn(2)
    edges = torch.tensor([[0], [1]])
    expected = actor(nodes, edges, image, scalar)
    padded = torch.cat([nodes, torch.randn(1, 2)], dim=0)
    actual = actor(padded, edges, image, scalar, node_mask=torch.tensor([1, 1, 0]))
    assert torch.allclose(expected, actual, atol=1e-6)


def test_missing_views_and_scalar_features_do_not_affect_logits():
    torch.manual_seed(3)
    actor = StageActor(2, 2, 1, 3, 8)
    nodes = torch.randn(2, 3, 2)
    edges = torch.tensor([[0], [1]])
    image = torch.randn(2, 1, 4, 4)
    scalars = torch.randn(2, 2)
    view_mask = torch.tensor([[1, 0, 1], [1, 0, 1]])
    scalar_mask = torch.tensor([[1, 0], [1, 0]])
    expected = actor(nodes, edges, image, scalars, view_mask=view_mask, scalar_mask=scalar_mask)
    scalars[:, 1] = 999
    actual = actor(nodes, edges, image * 100, scalars, view_mask=view_mask, scalar_mask=scalar_mask)
    assert torch.allclose(expected, actual)
