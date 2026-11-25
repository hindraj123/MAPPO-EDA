"""CPU multiprocessing rollout collection for MAPPO.

Workers own environments only.  Policy inference stays in the parent process,
so a CUDA policy is never serialized or initialized in a worker.  The policy
version is attached to every transition to enforce strictly on-policy data.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import time
import traceback
from typing import Any, Callable, Protocol, Sequence


class BatchedPolicy(Protocol):
    """Main-process interface for batched action inference.

    Implementations may run on CPU or CUDA in the parent process.  Workers
    never receive this object.
    """

    def act_batch(
        self, observations: Sequence[Any], *, policy_version: str
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class ObservationBatch:
    observations: tuple[Any, ...]
    worker_ids: tuple[int, ...]
    step: int
    policy_version: str


@dataclass(frozen=True)
class Transition:
    worker_id: int
    step: int
    observation: Any
    action: Any
    reward: Any
    next_observation: Any
    terminated: bool
    truncated: bool
    info: Any
    policy_version: str
    workdir: str

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class EpisodeTrajectory:
    worker_id: int
    transitions: tuple[Transition, ...]
    policy_version: str
    workdir: str


@dataclass(frozen=True)
class WorkerFailure:
    worker_id: int
    phase: str
    error_type: str
    message: str
    traceback: str
    workdir: str
    timed_out: bool = False


@dataclass(frozen=True)
class RolloutCollectionResult:
    trajectories: tuple[EpisodeTrajectory, ...]
    failures: tuple[WorkerFailure, ...]
    policy_version: str
    worker_count: int
    shutdown_clean: bool


class RolloutCollectionError(RuntimeError):
    """Raised when strict collection cannot produce a complete batch."""

    def __init__(self, result: RolloutCollectionResult) -> None:
        self.result = result
        detail = "; ".join(
            f"worker {f.worker_id} {f.phase}: {f.message}" for f in result.failures
        )
        super().__init__(f"rollout collection failed: {detail or 'unknown failure'}")


EnvFactory = Callable[[int, Path], Any]


def _normalize_reset(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return value[0]
    return value


def _normalize_step(value: Any) -> tuple[Any, Any, bool, bool, Any]:
    if not isinstance(value, tuple):
        raise TypeError("environment step must return a tuple")
    if len(value) == 5:
        observation, reward, terminated, truncated, info = value
    elif len(value) == 4:
        observation, reward, done, info = value
        terminated, truncated = bool(done), False
    else:
        raise ValueError("environment step must return 4 or 5 values")
    return observation, reward, bool(terminated), bool(truncated), info


def _write_failure(workdir: Path, text: str) -> None:
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "worker.failure.log").write_text(text, encoding="utf-8")
    except OSError:
        pass


def _worker_main(
    worker_id: int,
    env_factory: EnvFactory,
    workdir: str,
    commands: Any,
    results: Any,
) -> None:
    # Workers are CPU-only and do not import torch or construct a policy.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = Path(workdir)
    env = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        env = env_factory(worker_id, path)
        observation = _normalize_reset(env.reset())
        results.put(("ready", worker_id, observation, str(path)))
        while True:
            message = commands.get()
            kind = message[0]
            if kind == "stop":
                results.put(("stopped", worker_id))
                return
            if kind == "reset":
                observation = _normalize_reset(env.reset())
                results.put(("reset", worker_id, observation))
                continue
            if kind != "step":
                raise ValueError(f"unknown worker command: {kind!r}")
            _, step, action = message
            next_observation, reward, terminated, truncated, info = _normalize_step(
                env.step(action)
            )
            results.put(
                (
                    "transition",
                    worker_id,
                    step,
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
            )
    except BaseException as exc:  # transport failures to the parent process
        text = traceback.format_exc()
        _write_failure(path, text)
        results.put(("failure", worker_id, type(exc).__name__, str(exc), text, str(path)))
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


class MultiprocessRolloutCollector:
    """Collect fixed-horizon rollouts with one independent environment/worker."""

    def __init__(
        self,
        env_factory: EnvFactory,
        policy: BatchedPolicy,
        *,
        num_workers: int,
        horizon: int,
        policy_version: str,
        worker_timeout_s: float = 1800.0,
        work_root: Path | str | None = None,
        keep_workdirs: bool = False,
        start_method: str = "spawn",
    ) -> None:
        if num_workers <= 0 or horizon <= 0:
            raise ValueError("num_workers and horizon must be positive")
        if worker_timeout_s <= 0:
            raise ValueError("worker_timeout_s must be positive")
        if not policy_version:
            raise ValueError("policy_version must be non-empty")
        self.env_factory = env_factory
        self.policy = policy
        self.num_workers = num_workers
        self.horizon = horizon
        self.policy_version = policy_version
        self.worker_timeout_s = worker_timeout_s
        self.work_root = Path(work_root).resolve() if work_root else None
        self.keep_workdirs = keep_workdirs
        self.start_method = start_method

    def collect(self, *, strict: bool = True) -> RolloutCollectionResult:
        context = mp.get_context(self.start_method)
        temporary: tempfile.TemporaryDirectory[str] | None = None
        root = self.work_root
        if root is None:
            temporary = tempfile.TemporaryDirectory(prefix="mappo-eda-rollouts-")
            root = Path(temporary.name)
        root.mkdir(parents=True, exist_ok=True)
        commands = [context.Queue() for _ in range(self.num_workers)]
        results = context.Queue()
        processes: list[mp.Process] = []
        failures: list[WorkerFailure] = []
        trajectories: list[list[Transition]] = [[] for _ in range(self.num_workers)]
        workdirs = [root / f"worker-{i}" for i in range(self.num_workers)]
        active: set[int] = set(range(self.num_workers))
        observations: dict[int, Any] = {}
        done: set[int] = set()
        shutdown_clean = True
        try:
            for worker_id in range(self.num_workers):
                process = context.Process(
                    target=_worker_main,
                    args=(worker_id, self.env_factory, str(workdirs[worker_id]), commands[worker_id], results),
                    name=f"mappo-rollout-{worker_id}",
                )
                process.start()
                processes.append(process)
            ready = self._wait_for(results, active, {"ready", "failure"}, "startup", workdirs, failures)
            for worker_id, message in ready.items():
                observations[worker_id] = message[2]
            active -= {f.worker_id for f in failures}
            if not active:
                return self._finish(trajectories, failures, processes, commands, results, temporary, root, strict, False)

            for step in range(self.horizon):
                reset_ids = active & done
                for worker_id in reset_ids:
                    commands[worker_id].put(("reset",))
                if reset_ids:
                    reset_messages = self._wait_for(results, reset_ids, {"reset", "failure"}, "reset", workdirs, failures)
                    for worker_id, message in reset_messages.items():
                        observations[worker_id] = message[2]
                    failed = {f.worker_id for f in failures}
                    active -= failed
                    done -= failed
                    done -= reset_ids - set(reset_messages)
                if not active:
                    break

                ids = tuple(sorted(active))
                batch = tuple(observations[i] for i in ids)
                version = self._current_policy_version()
                if version != self.policy_version:
                    raise ValueError(f"policy version changed during rollout: {version!r}")
                actions = self.policy.act_batch(batch, policy_version=self.policy_version)
                if len(actions) != len(ids):
                    raise ValueError(f"policy returned {len(actions)} actions for {len(ids)} observations")
                for worker_id, action in zip(ids, actions):
                    commands[worker_id].put(("step", step, action))
                messages = self._wait_for(results, set(ids), {"transition", "failure"}, "step", workdirs, failures)
                for worker_id, message in messages.items():
                    _, _, returned_step, next_observation, reward, terminated, truncated, info = message
                    trajectories[worker_id].append(Transition(worker_id, returned_step, observations[worker_id], actions[ids.index(worker_id)], reward, next_observation, terminated, truncated, info, self.policy_version, str(workdirs[worker_id])))
                    observations[worker_id] = next_observation
                    if terminated or truncated:
                        done.add(worker_id)
                failed = {f.worker_id for f in failures}
                active -= failed
                done -= failed
            result = self._finish(trajectories, failures, processes, commands, results, temporary, root, strict, shutdown_clean)
            return result
        except BaseException:
            shutdown_clean = self._shutdown(processes, commands, results, active, workdirs, failures)
            if temporary is not None:
                temporary.cleanup()
            raise

    def _current_policy_version(self) -> str:
        value = getattr(self.policy, "policy_version", self.policy_version)
        return str(value)

    def _wait_for(self, results, expected: set[int], kinds: set[str], phase: str, workdirs, failures):
        pending = set(expected)
        messages = {}
        deadline = time.monotonic() + self.worker_timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for worker_id in pending:
                    failures.append(WorkerFailure(worker_id, phase, "TimeoutError", "worker response timed out", "", str(workdirs[worker_id]), True))
                break
            try:
                message = results.get(timeout=remaining)
            except queue.Empty:
                for worker_id in pending:
                    failures.append(WorkerFailure(worker_id, phase, "TimeoutError", "worker response timed out", "", str(workdirs[worker_id]), True))
                break
            worker_id = message[1]
            if worker_id not in pending:
                continue
            if message[0] == "failure":
                failures.append(WorkerFailure(worker_id, phase, message[2], message[3], message[4], message[5]))
                pending.remove(worker_id)
            elif message[0] in kinds:
                messages[worker_id] = message
                pending.remove(worker_id)
        return messages

    def _shutdown(self, processes, commands, results, active, workdirs, failures) -> bool:
        for worker_id, process in enumerate(processes):
            if process.is_alive():
                try:
                    commands[worker_id].put(("stop",))
                except Exception:
                    pass
        clean = True
        for process in processes:
            process.join(timeout=min(self.worker_timeout_s, 5.0))
        for process in processes:
            if process.is_alive():
                clean = False
                process.terminate()
                process.join(timeout=2.0)
        return clean

    def _finish(self, trajectories, failures, processes, commands, results, temporary, root, strict, shutdown_clean):
        shutdown_clean = self._shutdown(processes, commands, results, set(), [], failures) and shutdown_clean
        result = RolloutCollectionResult(tuple(EpisodeTrajectory(i, tuple(t), self.policy_version, str(root / f"worker-{i}")) for i, t in enumerate(trajectories) if t), tuple(failures), self.policy_version, self.num_workers, shutdown_clean)
        if temporary is not None and not self.keep_workdirs:
            temporary.cleanup()
        if strict and result.failures:
            raise RolloutCollectionError(result)
        return result
