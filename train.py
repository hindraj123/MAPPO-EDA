"""MAPPO-EDA training entry point with explicit synthetic and real modes."""
from __future__ import annotations
import argparse, json, random, shutil, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from mappo_eda.checkpoint import load_checkpoint, save_checkpoint, preserve_best_gdsii
from mappo_eda.evaluation import build_policy
from mappo_eda.logging import RunLogger
from mappo_eda.rl.buffer import OnPolicyRolloutBuffer, RolloutBatch
from mappo_eda.rl.policy import MAPPOPolicy
from mappo_eda.rl.trainer import MAPPOTrainer
from mappo_eda.rollout.collector import MultiprocessRolloutCollector
from mappo_eda.rollout.real import OpenROADWorkerFactory, ParentPolicy, actor_inputs, _stage
from mappo_eda.telemetry import start_wandb, system_metrics, physical_metrics, public_config


class _SyntheticCollectorEnv:
    """Synthetic worker fixture; it produces no OpenROAD measurements."""
    def __init__(self, worker_id, workdir): self.worker_id, self.workdir, self.step_no = worker_id, workdir, 0
    def reset(self): self.step_no = 0; return {"worker_id": self.worker_id, "synthetic_fixture": True}
    def step(self, action):
        self.step_no += 1
        done = self.step_no >= 4
        observation = {"worker_id": self.worker_id, "synthetic_fixture": True}
        return observation, 0.0, done, False, {"synthetic_fixture": True}
    def close(self): pass


class _SyntheticParentPolicy:
    def act_batch(self, observations, *, policy_version):
        return [0 for _ in observations]

def _read_config(path: Path) -> dict[str, Any]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line: continue
        key, value = (x.strip() for x in line.split(":", 1))
        if value.lower() in {"true", "false"}: result[key] = value.lower() == "true"
        else:
            try: result[key] = float(value) if "." in value else int(value)
            except ValueError: result[key] = value.strip("'\"")
    return result

def _seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def _episode(policy, cfg, seed):
    device = next(policy.parameters()).device; g = torch.Generator(device=device).manual_seed(seed); out = []
    for i, stage in enumerate(MAPPOPolicy.STAGES):
        nodes = torch.randn(1, 5, int(cfg["node_dim"]), generator=g, device=device); image = torch.randn(1, int(cfg["image_channels"]), 8, 8, generator=g, device=device); scalars = torch.randn(1, int(cfg["scalar_dim"]), generator=g, device=device); global_state = torch.randn(1, int(cfg["global_dim"]), generator=g, device=device); edges = torch.tensor([[0,1,2,3],[1,2,3,4]], device=device)
        mask = torch.zeros(1, int(cfg["action_dim"]), dtype=torch.bool, device=device); mask[:, :max(1, int(cfg["action_dim"])-i)] = True
        action, logp, _ = policy.actor(stage).act(nodes, edges, image, scalars, mask); value = policy.value(global_state); reward = torch.tensor([1.0 - abs(float(action.item()) - ((i + 1) % int(cfg["action_dim"]))) / max(1, int(cfg["action_dim"]))], device=device)
        out.append({"stage":stage,"nodes":nodes.cpu(),"edges":edges.cpu(),"image":image.cpu(),"scalars":scalars.cpu(),"global_state":global_state.cpu(),"action_mask":mask.cpu(),"action":action.cpu(),"log_prob":logp.detach().cpu(),"value":value.detach().cpu(),"reward":reward.cpu(),"done":torch.tensor([i == 3]),"synthetic_fixture":True})
    return out

def _worker(args):
    _, cfg, state, seed = args; policy = build_policy(cfg, "cpu"); policy.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()}); policy.eval()
    transitions = [t for e in range(int(cfg.get("episodes_per_worker", 1))) for t in _episode(policy, cfg, seed + e)]
    return [{k: (v.detach().cpu().numpy().copy() if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in transitions]

def collect_rollouts(policy, cfg, seed):
    workers = max(1, int(cfg.get("rollout_workers", 1)))
    collected = MultiprocessRolloutCollector(_SyntheticCollectorEnv, _SyntheticParentPolicy(), num_workers=workers, horizon=int(cfg.get("rollout_steps", 8)), policy_version=f"policy-{seed}").collect()
    if collected.failures:
        raise RuntimeError("synthetic rollout worker failure: " + "; ".join(f.message for f in collected.failures))
    # Tensor construction and policy optimization remain in the parent process.
    return [t for episode in collected.trajectories for t in _episode(policy, cfg, seed + episode.worker_id)]

def _real_experiment(cfg):
    path = cfg.get("openroad_experiment")
    if not path: raise RuntimeError("Real OpenROAD training requires openroad_experiment JSON with deployment-specific inputs; no synthetic fallback is used.")
    experiment = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("design", "initialization_script", "extraction_script", "stage_scripts", "stage_actions", "reward_scales", "reward_directions")
    missing = [key for key in required if not experiment.get(key)]
    if missing: raise ValueError("missing real experiment inputs: " + ", ".join(missing))
    for stage in MAPPOPolicy.STAGES:
        if not experiment["stage_scripts"].get(stage) or not experiment["stage_actions"].get(stage):
            raise ValueError(f"missing stage script or action catalog for {stage}")
    paths = [experiment["initialization_script"], experiment["extraction_script"], *experiment["stage_scripts"].values()]
    for item in paths:
        if not Path(item).is_file(): raise FileNotFoundError(f"OpenROAD experiment script unavailable: {item}")
    if set(experiment["reward_scales"]) != set(experiment["reward_directions"]):
        raise ValueError("reward scales and directions must cover the same measured metrics")
    return experiment

def _train_real(cfg, run_dir):
    executable = str(cfg.get("openroad_executable", "openroad"))
    if shutil.which(executable) is None: raise RuntimeError(f"Real OpenROAD training requested, but executable {executable!r} is unavailable; no synthetic fallback is used.")
    experiment = _real_experiment(cfg)
    _seed(int(cfg.get("seed", 7)))
    device = torch.device(str(cfg.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable in this session")
    policy = build_policy(cfg, device)
    trainer = MAPPOTrainer(policy, lr=float(cfg["learning_rate"]), clip=float(cfg["ppo_clip"]), device=device)
    start_update = 1
    checkpoint = None
    if cfg.get("resume_checkpoint"):
        checkpoint = load_checkpoint(cfg["resume_checkpoint"], policy=policy, optimizer=trainer.optimizer,
                                     map_location=device, restore_rng=True)
        if checkpoint.get("metadata", {}).get("synthetic_fixture") is not False:
            raise ValueError("real training can only resume an explicitly real training checkpoint")
        if checkpoint["metadata"].get("experiment") != public_config(experiment):
            raise ValueError("checkpoint experiment does not match the requested real experiment")
        start_update = int(checkpoint["progress"]["update"]) + 1
    logger = RunLogger(run_dir)
    (run_dir / "config.json").write_text(json.dumps(public_config(cfg), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "openroad-experiment.json").write_text(json.dumps(public_config(experiment), indent=2, sort_keys=True), encoding="utf-8")
    logger.event(kind="run_started", mode="real_openroad", seed=int(cfg.get("seed", 7)))
    wandb_run = start_wandb(cfg, run_dir, checkpoint.get("metadata", {}).get("wandb_run_id") if checkpoint else None) if cfg.get("wandb_enabled", True) else None
    try:
        return _real_updates(cfg, run_dir, experiment, policy, trainer, logger, device, start_update, checkpoint, wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

def _real_updates(cfg, run_dir, experiment, policy, trainer, logger, device, start_update, checkpoint, wandb_run):
    episodes = int(checkpoint.get("progress", {}).get("episodes", 0)) if checkpoint else 0
    best_score = checkpoint.get("best_metric") if checkpoint else None
    started = time.monotonic()
    for update in range(start_update, int(cfg["updates"]) + 1):
        # Each collection starts fresh worker environments. Do not combine a
        # horizon-cut fragment with the next collection's unrelated episode.
        partial = {}
        version = f"policy-{update}"
        collector = MultiprocessRolloutCollector(OpenROADWorkerFactory(cfg, experiment), ParentPolicy(policy, version),
            num_workers=int(cfg.get("rollout_workers", 4)), horizon=int(cfg.get("rollout_steps", 4)), policy_version=version,
            work_root=run_dir / "workers" / f"update-{update:05d}", keep_workdirs=True,
            worker_timeout_s=float(cfg.get("worker_timeout_s", 1800)))
        try: result = collector.collect()
        except Exception as exc:
            logger.failure(kind="rollout_failed", update=update, error=str(exc))
            raise
        if not any(t.info.get("reward_inputs", {}).get("normalized_deltas") for trajectory in result.trajectories for t in trajectory.transitions if t.info.get("reward_inputs")):
            raise RuntimeError("OpenROAD rollout has no measured normalized reward deltas; refusing optimization")
        records = []
        completed = []
        for trajectory in result.trajectories:
            transitions = trajectory.transitions
            buf = OnPolicyRolloutBuffer(len(transitions), 1, 1, device=device)
            for t in transitions:
                buf.add(observations=torch.zeros(()), actions=t.action["action"], log_probs=t.action["log_prob"],
                        rewards=t.reward, values=t.action["value"], dones=t.done)
                logger.event(kind="stage", update=update, worker=trajectory.worker_id, stage=t.info["stage"],
                             metrics=t.info["metrics"], reward=t.reward, workdir=t.workdir, policy_version=version)
                episode = partial.setdefault(trajectory.worker_id, {"reward": 0., "stage_rewards": {}, "duration_s": 0.})
                episode["reward"] += float(t.reward)
                episode["stage_rewards"][t.info["stage"]] = float(t.reward)
                episode["duration_s"] += float(t.info.get("duration_s", 0.))
                if t.done:
                    episode.update(metrics=t.info.get("metrics", {}), succeeded=bool(t.info.get("succeeded")) and t.terminated and not t.truncated, workdir=t.info.get("workdir"), runtime_s=t.info.get("episode_runtime_s"))
                    completed.append(episode)
                    del partial[trajectory.worker_id]
            last = transitions[-1]
            if last.done: bootstrap = torch.tensor(0., device=device)
            else:
                with torch.no_grad(): bootstrap = policy.value(torch.as_tensor(last.next_observation.global_state, device=device))
            batch = buf.compute_gae(bootstrap, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            records.extend((t, batch.advantages[i], batch.returns[i]) for i, t in enumerate(transitions))
        tracked = {}
        for stage in MAPPOPolicy.STAGES:
            selected = [row for row in records if _stage(row[0].observation) == stage]
            if not selected: continue
            ts = [row[0] for row in selected]
            old = torch.tensor([t.action["log_prob"] for t in ts], device=device)
            actions = torch.tensor([t.action["action"] for t in ts], device=device)
            advantages = torch.stack([row[1] for row in selected]); returns = torch.stack([row[2] for row in selected])
            values = torch.tensor([t.action["value"] for t in ts], device=device)
            batch = RolloutBatch(torch.zeros_like(values), actions, old, torch.zeros_like(values), values, returns, advantages, torch.ones_like(values), torch.zeros_like(values))
            inputs = actor_inputs([t.observation for t in ts], device)
            states = torch.as_tensor(np.stack([t.observation.global_state for t in ts]), device=device)
            stats = trainer.update(batch=batch, stage=stage, actor_inputs=inputs, global_state=states, epochs=int(cfg.get("ppo_epochs", 1)))
            logger.event(kind="stage_update", update=update, stage=stage, **stats)
            for key, value in stats.items(): tracked[f"training/{stage}/{key}"] = value
        previous_episodes = episodes
        episodes += len(completed)
        if completed:
            tracked["training/mean_episode_reward"] = sum(x["reward"] for x in completed) / len(completed)
            tracked["system/episode_tool_runtime_seconds"] = sum(x["duration_s"] for x in completed) / len(completed)
            runtimes = [x["runtime_s"] for x in completed if x["runtime_s"] is not None]
            if runtimes: tracked["system/episode_runtime_seconds"] = sum(runtimes) / len(runtimes)
            tracked["system/episodes_per_hour"] = 3600 * (episodes - (int(checkpoint.get("progress", {}).get("episodes", 0)) if checkpoint else 0)) / max(time.monotonic() - started, 1e-9)
            for stage in MAPPOPolicy.STAGES:
                rewards = [x["stage_rewards"][stage] for x in completed if stage in x["stage_rewards"]]
                if rewards: tracked[f"training/{stage}/reward"] = sum(rewards) / len(rewards)
            # A completed flow is valid only when DRC is explicitly measured.
            assessed = [x for x in completed if x["succeeded"] and "drc_violations" in x["metrics"]]
            if len(assessed) == len(completed): tracked["physical/valid_design_rate"] = sum(x["metrics"]["drc_violations"] == 0 for x in assessed) / len(completed)
            for episode in completed:
                if episode["succeeded"]:
                    tracked.update(physical_metrics(episode["metrics"], experiment.get("baseline_metrics"), experiment.get("reward_directions")))
                    key = experiment.get("best_metric")
                    gds_name = experiment.get("gds_filename")
                    if key and gds_name and episode["metrics"].get("drc_violations") == 0 and key in episode["metrics"]:
                        direction = experiment.get("reward_directions", {}).get(key)
                        if direction not in (-1, 1): raise ValueError("best_metric requires a reward direction")
                        score = direction * episode["metrics"][key]
                        root = Path(episode["workdir"]).resolve(); source = (root / gds_name).resolve()
                        if not source.is_relative_to(root): raise ValueError("gds_filename must stay inside the episode directory")
                        if source.is_file() and source.stat().st_size and (best_score is None or score > best_score):
                            preserve_best_gdsii(source, run_dir / "best-valid.gds"); best_score = score
        tracked.update(system_metrics(run_dir, device))
        tracked["training/episodes"] = episodes
        if wandb_run is not None: wandb_run.log(tracked, step=update)
        logger.event(kind="update", update=update, transitions=len(records), policy_version=version)
        checkpoint_every = int(cfg.get("checkpoint_episodes", 50))
        if checkpoint_every <= 0: raise ValueError("checkpoint_episodes must be positive")
        if episodes // checkpoint_every > previous_episodes // checkpoint_every:
            save_checkpoint(run_dir / f"checkpoint-{update:05d}.pt", policy=policy, optimizer=trainer.optimizer,
                            config=public_config(cfg), progress={"update": update, "transitions": len(records), "episodes": episodes}, seed=int(cfg.get("seed", 7)),
                            policy_version=update, best_metric=best_score, metadata={"synthetic_fixture": False, "experiment": public_config(experiment), "wandb_run_id": wandb_run.id if wandb_run else None})
    logger.event(kind="run_finished", status="complete")
    return run_dir

def train(cfg, run_dir: Path, synthetic=True):
    if not synthetic:
        return _train_real(cfg, run_dir)
    _seed(int(cfg.get("seed",7))); device = torch.device(str(cfg.get("device","cpu")))
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    policy = build_policy(cfg, device); trainer = MAPPOTrainer(policy, lr=float(cfg["learning_rate"]), clip=float(cfg["ppo_clip"]), device=device); logger = RunLogger(run_dir); run_dir.mkdir(parents=True, exist_ok=True); (run_dir/"config.json").write_text(json.dumps(cfg,indent=2,sort_keys=True),encoding="utf-8"); logger.event(kind="run_started",mode="synthetic_development",synthetic_fixture=True,seed=int(cfg.get("seed",7)),real_openroad_verification="pending")
    for update in range(1,int(cfg["updates"])+1):
        transitions = collect_rollouts(policy,cfg,int(cfg.get("seed",7))+update*1000); buffer = OnPolicyRolloutBuffer(len(transitions),1,1,device=device)
        for t in transitions: buffer.add(observations=torch.zeros(1),actions=t["action"],log_probs=t["log_prob"],rewards=t["reward"],values=t["value"],dones=t["done"])
        batch = buffer.compute_gae(torch.zeros(1,device=device),float(cfg["gamma"]),float(cfg["gae_lambda"]))
        for stage in MAPPOPolicy.STAGES:
            selected=[i for i,t in enumerate(transitions) if t["stage"]==stage]
            if not selected: continue
            inputs={k:(transitions[selected[0]][k].to(device) if k == "edges" else torch.cat([transitions[i][k].to(device) for i in selected])) for k in ("nodes","edges","image","scalars","action_mask")}; actions=torch.cat([transitions[i]["action"].to(device) for i in selected]); new_lp,entropy=trainer.evaluate_actor(stage,inputs,actions); old_lp=torch.cat([transitions[i]["log_prob"].to(device) for i in selected]); vals=policy.value(torch.cat([transitions[i]["global_state"].to(device) for i in selected])); idx=torch.tensor(selected,device=device); trainer.update(new_lp,old_lp,batch.advantages[idx].reshape(-1),vals,batch.returns[idx].reshape(-1),entropy=entropy)
        logger.event(kind="update",update=update,synthetic_fixture=True,transitions=len(transitions));
        if update % int(cfg.get("checkpoint_interval",1)) == 0: save_checkpoint(run_dir/f"checkpoint-{update:05d}.pt",policy=policy,optimizer=trainer.optimizer,config=cfg,progress={"update":update,"transitions":len(transitions)},seed=int(cfg.get("seed",7)),metadata={"synthetic_fixture":True})
    logger.event(kind="run_finished",status="synthetic_complete",real_openroad_verification="pending"); return run_dir

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("configs/training.yaml")); p.add_argument("--run-dir",type=Path); p.add_argument("--device"); p.add_argument("--updates",type=int); p.add_argument("--steps",type=int); p.add_argument("--resume",type=Path); p.add_argument("--synthetic",action="store_true"); a=p.parse_args(); cfg=_read_config(a.config)
    if a.run_dir: cfg["run_dir"]=str(a.run_dir)
    if a.device: cfg["device"]=a.device
    if a.updates: cfg["updates"]=a.updates
    if a.steps: cfg["rollout_steps"]=a.steps
    if a.resume:
        if a.synthetic: p.error("--resume is supported for real training only")
        cfg["resume_checkpoint"] = str(a.resume)
    train(cfg,Path(cfg["run_dir"]),synthetic=a.synthetic)
if __name__ == "__main__": main()
