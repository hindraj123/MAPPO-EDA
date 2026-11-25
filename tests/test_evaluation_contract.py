"""Checks for the explicitly synthetic evaluation contract."""
import pytest
import torch

from evaluate import evaluate_checkpoint
from mappo_eda.evaluation import build_policy, evaluate_synthetic


@pytest.fixture
def config():
    return {"node_dim": 3, "scalar_dim": 2, "image_channels": 1,
            "action_dim": 4, "hidden_dim": 8, "global_dim": 5}


def test_synthetic_evaluation_is_labeled_and_seed_reproducible(config):
    torch.manual_seed(1)
    policy = build_policy(config)
    first = evaluate_synthetic(policy, config, episodes=2, seed=13)
    second = evaluate_synthetic(policy, config, episodes=2, seed=13)
    assert first == second
    assert first["mode"] == "synthetic"
    assert first["synthetic_fixture"] is True
    assert set(first["episodes"]) == {"policy", "first_valid", "last_valid", "random"}
    assert all(episode["synthetic_fixture"] for episodes in first["episodes"].values() for episode in episodes)


def test_real_checkpoint_never_falls_back_to_synthetic(tmp_path, config):
    policy = build_policy(config)
    checkpoint = tmp_path / "real.pt"
    torch.save({"config": config, "policy_state_dict": policy.state_dict(), "metadata": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="no synthetic fallback"):
        evaluate_checkpoint(checkpoint)
