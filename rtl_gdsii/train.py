"""MAPPO-EDA training entry point with explicit synthetic and real modes."""
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path
from typing import Any
import numpy as np
import torch
from mappo_eda.checkpoint import save_checkpoint
from mappo_eda.evaluation import build_policy
from mappo_eda.logging import RunLogger
from mappo_eda.rl.buffer import OnPolicyRolloutBuffer
from mappo_eda.rl.policy import MAPPOPolicy
from mappo_eda.rl.trainer import MAPPOTrainer
from mappo_eda.rollout.collector import MultiprocessRolloutCollector


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
    g = torch.Generator().manual_seed(seed); device = next(policy.parameters()).device; out = []
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

def train(cfg, run_dir: Path, synthetic=True):
    if not synthetic:
        executable = str(cfg.get("openroad_executable", "openroad"))
        if shutil.which(executable) is None: raise RuntimeError(f"Real OpenROAD training requested, but executable {executable!r} is unavailable; no synthetic fallback is used.")
        raise RuntimeError("Real OpenROAD training requires deployment-specific technology inputs; no synthetic fallback is used.")
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
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("configs/training.yaml")); p.add_argument("--run-dir",type=Path); p.add_argument("--device"); p.add_argument("--updates",type=int); p.add_argument("--steps",type=int); p.add_argument("--synthetic",action="store_true"); a=p.parse_args(); cfg=_read_config(a.config)
    if a.run_dir: cfg["run_dir"]=str(a.run_dir)
    if a.device: cfg["device"]=a.device
    if a.updates: cfg["updates"]=a.updates
    train(cfg,Path(cfg["run_dir"]),synthetic=a.synthetic)
if __name__ == "__main__": main()
