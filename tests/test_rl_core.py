"""Synthetic RL-core tests; no OpenROAD metrics or trajectories are used."""

import torch
import pytest

from mappo_eda.rl.buffer import OnPolicyRolloutBuffer
from mappo_eda.rl.encoders import CNNEncoder, GraphEncoder, MLPEncoder
from mappo_eda.rl.losses import ppo_loss
from mappo_eda.rl.policy import MAPPOPolicy
from mappo_eda.rl.reward import weighted_reward
from mappo_eda.rl.trainer import MAPPOTrainer


def test_encoders_support_batched_tensors():
    assert MLPEncoder(3, 8)(torch.randn(4, 3)).shape == (4, 8)
    assert CNNEncoder(1, 8)(torch.randn(4, 1, 6, 6)).shape == (4, 8)
    nodes = torch.randn(4, 5, 3)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert GraphEncoder(3, 8)(nodes, edges).shape == (4, 5, 8)


def test_policy_has_four_actors_and_masks_actions():
    policy = MAPPOPolicy({"node_dim": 4, "scalar_dim": 3, "image_channels": 1, "action_dim": 5}, global_dim=8, hidden_dim=16)
    assert set(policy.actors) == {"floorplanning", "placement", "cts", "routing"}
    inputs = (torch.randn(2, 6, 4), torch.tensor([[0, 1], [1, 2]]), torch.randn(2, 1, 5, 5), torch.randn(2, 3))
    mask = torch.tensor([[True, False, False, True, False], [False, True, False, False, True]])
    actions, log_probs, entropy = policy.actor("routing").act(*inputs, action_mask=mask)
    assert actions.shape == log_probs.shape == entropy.shape == (2,)
    assert torch.all(mask.gather(1, actions[:, None]).squeeze(1))


def test_gae_respects_terminal_and_is_on_policy():
    buffer = OnPolicyRolloutBuffer(2, 1, 1)
    for reward, done in ((1.0, 0.0), (2.0, 1.0)):
        buffer.add(observations=torch.zeros(1), actions=torch.zeros(1, dtype=torch.long), log_probs=torch.zeros(1), rewards=torch.tensor([reward]), values=torch.zeros(1), dones=torch.tensor([done]))
    batch = buffer.compute_gae(torch.tensor([10.0]))
    assert torch.allclose(batch.advantages, torch.tensor([[1.0 + 0.99 * 0.95 * 2.0], [2.0]]))
    buffer.clear()
    assert len(buffer) == 0


def test_ppo_singleton_is_finite_and_reward_requires_measured_fields():
    assert torch.isfinite(ppo_loss(torch.zeros(1), torch.zeros(1), torch.ones(1)))
    assert weighted_reward({"area": 2.0, "wns": 1.0}, {"area": 1.0, "wns": 2.0}) == -4.0


def test_trainer_can_backpropagate_through_actor_and_critic_on_cpu():
    policy = MAPPOPolicy({"node_dim": 3, "scalar_dim": 2, "image_channels": 1, "action_dim": 4}, global_dim=6, hidden_dim=8)
    trainer = MAPPOTrainer(policy)
    inputs = {"nodes": torch.randn(2, 4, 3), "edges": torch.tensor([[0, 1], [1, 2]]), "image": torch.randn(2, 1, 4, 4), "scalars": torch.randn(2, 2)}
    actions, old_log_probs, _ = policy.actor("cts").act(**inputs)
    new_log_probs, _ = trainer.evaluate_actor("cts", inputs, actions)
    values = policy.value(torch.randn(2, 6))
    stats = trainer.update(new_log_probs, old_log_probs.detach(), torch.ones(2), values, torch.ones(2))
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_cuda_path_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this session")
    device = torch.device("cuda")
    encoder = MLPEncoder(3, 8).to(device)
    result_device = encoder(torch.randn(2, 3, device=device)).device
    assert result_device.type == "cuda"
    assert result_device.index == torch.cuda.current_device()
