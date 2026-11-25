"""Explicit OpenROAD worker adapter and parent-only batched policy inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import time
import numpy as np
import torch

from mappo_eda.environment.openroad import OpenROADCommandRunner
from mappo_eda.environment.workflow import OpenROADEnvironment
from mappo_eda.rl.policy import MAPPOPolicy


@dataclass
class OpenROADWorkerFactory:
    config: Mapping[str, Any]
    experiment: Mapping[str, Any]

    def __call__(self, worker_id: int, workdir: Path):
        cfg, exp = self.config, self.experiment
        env = OpenROADEnvironment(
            design=str(exp['design']), runner=OpenROADCommandRunner(str(cfg.get('openroad_executable', 'openroad')), threads=int(cfg.get('openroad_threads', 2))),
            workdir=workdir, stage_scripts=exp['stage_scripts'], initialization_script=exp['initialization_script'],
            extraction_script=exp['extraction_script'], stage_actions=exp['stage_actions'],
            reward_scales=exp['reward_scales'], reward_directions=exp['reward_directions'],
            required_environment=exp.get('required_environment', ()), action_dim=int(cfg['action_dim']),
            node_dim=int(cfg['node_dim']), scalar_dim=int(cfg['scalar_dim']),
            image_channels=int(cfg['image_channels']), global_dim=int(cfg['global_dim']), require_metrics=True,
        )
        env.require_openroad()
        return _WorkerEnvironment(env)


class _WorkerEnvironment:
    def __init__(self, env): self.env = env
    def reset(self):
        self.started_at = time.monotonic()
        self.env.reset()
        return self.env.observation()
    def step(self, payload):
        result = self.env.step(int(payload['action']))
        if not result.info['succeeded']:
            raise RuntimeError(f"OpenROAD stage {result.info['stage']} failed: {result.info['error']}")
        observation, reward, terminated, truncated, info = tuple(result)
        if terminated or truncated:
            info['episode_runtime_s'] = time.monotonic() - self.started_at
        return observation, reward, terminated, truncated, info
    def close(self): pass


def _stage(observation):
    return getattr(observation.stage, 'value', observation.stage)


def actor_inputs(observations, device):
    """Batch observations sharing one topology for a stage."""
    if not observations: raise ValueError('empty observation batch')
    edges = observations[0].edges
    if any(not np.array_equal(edges, obs.edges) for obs in observations[1:]):
        raise ValueError('batched graph topologies differ; cannot share edges')
    inputs = {name: torch.as_tensor(np.stack([getattr(obs, name) for obs in observations]), device=device)
              for name in ('nodes', 'image', 'scalars', 'node_mask', 'action_mask')}
    inputs['edges'] = torch.as_tensor(edges, device=device)
    return inputs


class ParentPolicy:
    def __init__(self, policy, policy_version):
        self.policy, self.policy_version = policy, policy_version
        self.device = next(policy.parameters()).device

    @torch.no_grad()
    def act_batch(self, observations, *, policy_version):
        if policy_version != self.policy_version: raise ValueError('policy version mismatch')
        output = [None] * len(observations)
        for stage in MAPPOPolicy.STAGES:
            indices = [i for i, observation in enumerate(observations) if _stage(observation) == stage]
            if not indices: continue
            selected = [observations[i] for i in indices]
            actions, log_probs, _ = self.policy.actor(stage).act(**actor_inputs(selected, self.device))
            global_states = torch.as_tensor(np.stack([x.global_state for x in selected]), device=self.device)
            values = self.policy.value(global_states)
            for j, index in enumerate(indices):
                output[index] = {'action': int(actions[j]), 'log_prob': float(log_probs[j]), 'value': float(values[j])}
        if any(item is None for item in output): raise ValueError('unknown observation stage')
        return output
