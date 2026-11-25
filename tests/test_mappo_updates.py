"""Synthetic checks of on-policy MAPPO optimization; no EDA measurements."""
import pytest
import torch

from mappo_eda.rl.buffer import RolloutBatch
from mappo_eda.rl.losses import ppo_loss
from mappo_eda.rl.policy import MAPPOPolicy
from mappo_eda.rl.trainer import MAPPOTrainer


def test_single_sample_keeps_policy_gradient():
    log_prob = torch.tensor([0.0], requires_grad=True)
    ppo_loss(log_prob, torch.zeros(1), torch.ones(1)).backward()
    assert log_prob.grad.item() == pytest.approx(-1.0)


def test_multiple_epochs_recompute_selected_actor_and_critic():
    torch.manual_seed(7)
    policy = MAPPOPolicy({"node_dim": 3, "scalar_dim": 2, "image_channels": 1,
                          "action_dim": 3, "hidden_dim": 8}, global_dim=4, hidden_dim=8)
    trainer = MAPPOTrainer(policy, entropy_coef=0.01)
    inputs = {"nodes": torch.randn(3, 4, 3), "edges": torch.tensor([[0, 1], [1, 2]]),
              "image": torch.randn(3, 1, 4, 4), "scalars": torch.randn(3, 2)}
    states = torch.randn(3, 4)
    with torch.no_grad():
        actions, log_probs, _ = policy.actor("cts").act(**inputs)
        values = policy.value(states)
    advantages = torch.tensor([-1., 0.5, 1.])
    batch = RolloutBatch(torch.zeros(3), actions, log_probs, torch.zeros(3), values,
                         values + advantages, advantages, torch.ones(3), torch.zeros(3))
    selected_before = policy.actor("cts").policy.weight.detach().clone()
    other_before = policy.actor("routing").policy.weight.detach().clone()
    critic_before = policy.critic.value[0].weight.detach().clone()
    stats = trainer.update(batch=batch, stage="cts", actor_inputs=inputs,
                           global_state=states, epochs=2)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert not torch.equal(selected_before, policy.actor("cts").policy.weight)
    assert torch.equal(other_before, policy.actor("routing").policy.weight)
    assert not torch.equal(critic_before, policy.critic.value[0].weight)
