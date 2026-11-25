"""Synthetic multiprocessing rollout tests; no OpenROAD metrics are used."""

from pathlib import Path
import time

import pytest

from mappo_eda.rollout import MultiprocessRolloutCollector, RolloutCollectionError


class SyntheticEnv:
    """Synthetic fixture only: deterministic finite environment per worker."""

    def __init__(self, worker_id, workdir, fail=False, hang=False):
        self.worker_id, self.workdir = worker_id, Path(workdir)
        self.fail, self.hang, self.step_count = fail, hang, 0
        (self.workdir / "owner.txt").write_text(str(worker_id), encoding="utf-8")

    def reset(self):
        self.step_count = 0
        return {"worker": self.worker_id, "step": 0, "action_mask": (True, False)}

    def step(self, action):
        if self.hang:
            time.sleep(0.5)
        if self.fail and self.step_count == 0:
            raise RuntimeError("synthetic worker failure")
        assert action == 0
        self.step_count += 1
        return ({"worker": self.worker_id, "step": self.step_count, "action_mask": (True, False)}, 1.0, self.step_count >= 2, {})

    def close(self):
        (self.workdir / "closed.txt").write_text("yes", encoding="utf-8")


class SyntheticFactory:
    def __init__(self, failing=(), hanging=()):
        self.failing, self.hanging = set(failing), set(hanging)

    def __call__(self, worker_id, workdir):
        return SyntheticEnv(worker_id, workdir, worker_id in self.failing, worker_id in self.hanging)


class SyntheticCPUPolicy:
    """Synthetic CPU batched policy fixture; inference remains in parent."""

    policy_version = "synthetic-v1"

    def __init__(self):
        self.batches = []

    def act_batch(self, observations, *, policy_version):
        assert policy_version == self.policy_version
        self.batches.append(tuple(observations))
        return [0] * len(observations)


def test_spawn_workers_are_isolated_and_use_batched_fixed_policy(tmp_path):
    policy = SyntheticCPUPolicy()
    result = MultiprocessRolloutCollector(SyntheticFactory(), policy, num_workers=2, horizon=3, policy_version="synthetic-v1", work_root=tmp_path / "workers", start_method="spawn").collect()
    assert not result.failures
    assert result.shutdown_clean
    assert len(policy.batches) == 3
    assert all(len(batch) == 2 for batch in policy.batches)
    paths = {trajectory.workdir for trajectory in result.trajectories}
    assert len(paths) == 2
    assert all(len(trajectory.transitions) == 3 for trajectory in result.trajectories)
    assert {t.policy_version for x in result.trajectories for t in x.transitions} == {"synthetic-v1"}
    assert all((Path(path) / "closed.txt").exists() for path in paths)


def test_worker_failure_can_return_partial_result_and_strict_mode_raises(tmp_path):
    policy = SyntheticCPUPolicy()
    result = MultiprocessRolloutCollector(SyntheticFactory(failing=(1,)), policy, num_workers=2, horizon=2, policy_version="synthetic-v1", work_root=tmp_path / "failure", start_method="spawn").collect(strict=False)
    assert any(f.worker_id == 1 and not f.timed_out for f in result.failures)
    with pytest.raises(RolloutCollectionError):
        MultiprocessRolloutCollector(SyntheticFactory(failing=(1,)), SyntheticCPUPolicy(), num_workers=2, horizon=2, policy_version="synthetic-v1", work_root=tmp_path / "strict", start_method="spawn").collect()


def test_worker_timeout_is_reported_and_shutdown(tmp_path):
    result = MultiprocessRolloutCollector(SyntheticFactory(hanging=(0,)), SyntheticCPUPolicy(), num_workers=1, horizon=1, policy_version="synthetic-v1", worker_timeout_s=0.1, work_root=tmp_path / "timeout", start_method="spawn").collect(strict=False)
    assert result.failures[0].timed_out
    assert not result.shutdown_clean
