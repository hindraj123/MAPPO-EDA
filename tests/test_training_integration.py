"""End-to-end synthetic integration tests; no physical metrics are claimed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from train import _read_config, train
from evaluate import evaluate_checkpoint
from mappo_eda.checkpoint import load_checkpoint
from mappo_eda.evaluation import build_policy
from mappo_eda.rl.trainer import MAPPOTrainer

def test_synthetic_rollout_update_checkpoint_and_evaluation(tmp_path: Path):
    cfg=_read_config(Path("configs/training.yaml")); cfg.update(updates=1,rollout_workers=2,episodes_per_worker=1,device="cpu")
    checkpoint=train(cfg,tmp_path/"run",synthetic=True)/"checkpoint-00001.pt"; result=evaluate_checkpoint(checkpoint,episodes=2)
    assert checkpoint.is_file() and result["synthetic_fixture"] and set(result["episodes"])=={"policy","first_valid","last_valid","random"}

def test_checkpoint_restores_optimizer_and_progress(tmp_path: Path):
    cfg=_read_config(Path("configs/training.yaml")); cfg.update(updates=1,rollout_workers=1,device="cpu"); run=train(cfg,tmp_path/"run",synthetic=True); policy=build_policy(cfg); trainer=MAPPOTrainer(policy); payload=load_checkpoint(run/"checkpoint-00001.pt",policy=policy,optimizer=trainer.optimizer)
    assert payload["progress"]["update"]==1 and trainer.optimizer.state_dict()["param_groups"]

def test_real_mode_fails_without_openroad(tmp_path: Path):
    cfg=_read_config(Path("configs/training.yaml")); cfg["openroad_executable"]="definitely-not-openroad"
    try: train(cfg,tmp_path/"run",synthetic=False)
    except RuntimeError as exc: assert "OpenROAD" in str(exc) and "synthetic fallback" in str(exc)
    else: raise AssertionError("real mode must not silently use synthetic data")
