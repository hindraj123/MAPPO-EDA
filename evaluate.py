"""Deterministic checkpoint evaluation and labeled synthetic baselines."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from mappo_eda.evaluation import evaluate_synthetic, restore_for_evaluation

def evaluate_checkpoint(checkpoint: Path, device="cpu", episodes=4, baselines=True):
    policy, cfg, payload = restore_for_evaluation(checkpoint, device)
    if payload.get("metadata", {}).get("synthetic_fixture", False):
        result = evaluate_synthetic(policy, cfg, episodes, int(payload.get("seed", 7)), baselines)
        result.update(checkpoint=str(checkpoint), update=payload.get("progress", {}).get("update"), real_openroad_verification="pending")
        return result
    raise RuntimeError("Real checkpoint evaluation requires the Ubuntu OpenROAD toolchain; no synthetic fallback is used.")

def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint",type=Path); p.add_argument("--device",default="cpu"); p.add_argument("--episodes",type=int,default=4); p.add_argument("--no-baselines",action="store_true"); p.add_argument("--output",type=Path); a=p.parse_args(); result=evaluate_checkpoint(a.checkpoint,a.device,a.episodes,not a.no_baselines); text=json.dumps(result,indent=2,sort_keys=True); print(text)
    if a.output: a.output.write_text(text+"\n",encoding="utf-8")
if __name__ == "__main__": main()
