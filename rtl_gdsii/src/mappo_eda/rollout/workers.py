"""CPU rollout workers; CUDA policy inference stays in the parent process."""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import tempfile
from typing import Any, Callable


@dataclass
class RolloutFailure:
    worker_id: int
    error: str
    timed_out: bool = False


def _worker_main(worker_id: int, factory: Callable[[Path], Any], steps: int, output: Any) -> None:
    workdir = Path(tempfile.mkdtemp(prefix=f"mappo-worker-{worker_id}-"))
    try:
        env = factory(workdir)
        trajectory = []
        observation = env.reset()
        for _ in range(steps):
            action = env.sample_action(observation)
            next_observation, reward, done, info = env.step(action)
            trajectory.append({"observation": observation, "action": action, "reward": reward, "done": done, "info": info})
            observation = env.reset() if done else next_observation
        output.put((worker_id, trajectory, None))
    except Exception as exc:  # worker errors are returned, never silently lost
        output.put((worker_id, None, repr(exc)))
    finally:
        close = getattr(locals().get("env"), "close", None)
        if close:
            close()


class RolloutCollector:
    def __init__(self, env_factory: Callable[[Path], Any], workers: int = 1, start_method: str = "spawn"):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.env_factory, self.workers = env_factory, workers
        self.context = mp.get_context(start_method)

    def collect(self, steps: int, timeout_s: float = 1800.0) -> tuple[list[list[dict[str, Any]]], list[RolloutFailure]]:
        if steps < 1 or timeout_s <= 0:
            raise ValueError("steps and timeout_s must be positive")
        output = self.context.Queue()
        processes = [self.context.Process(target=_worker_main, args=(i, self.env_factory, steps, output), daemon=True) for i in range(self.workers)]
        for process in processes:
            process.start()
        trajectories, failures = [], []
        try:
            for _ in processes:
                try:
                    worker_id, trajectory, error = output.get(timeout=timeout_s)
                except queue.Empty:
                    failures.extend(RolloutFailure(i, "worker timeout", True) for i in range(len(processes)))
                    break
                if error is not None:
                    failures.append(RolloutFailure(worker_id, error))
                else:
                    trajectories.append(trajectory)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
        return trajectories, failures


class SyntheticRolloutEnv:
    """Explicit synthetic fixture; never represents OpenROAD measurements."""
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.t = 0

    def reset(self):
        self.t = 0
        return {"stage": "floorplanning", "synthetic_fixture": True}

    def sample_action(self, observation):
        return 0

    def step(self, action):
        self.t += 1
        return {"stage": "floorplanning", "synthetic_fixture": True}, 0.0, self.t >= 2, {"synthetic_fixture": True}
