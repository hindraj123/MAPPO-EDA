"""Synthetic fake-tool tests of the real rollout adapter, not EDA measurements."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from train import _read_config, train


def test_synthetic_tool_rollout_reaches_stage_updates(tmp_path, monkeypatch):
    import train as training
    calls = []
    class FakeRun:
        id = 'test-run-id'
        def log(self, values, step): calls.append(('log', step, values))
        def finish(self): calls.append(('finish',))
    def fake_start(config, run_dir, run_id=None):
        calls.append(('start', run_id))
        return FakeRun()
    monkeypatch.setattr(training, 'start_wandb', fake_start)
    tool = tmp_path / 'synthetic-openroad'
    tool.write_text("#!/usr/bin/env python3\nimport os\ns = os.environ['MAPPO_EDA_STAGE']\nprint('MAPP0EDA_METRIC area_um2', {'floorplanning': 4, 'placement': 3, 'cts': 2, 'routing': 1}[s])\n")
    tool.chmod(0o700)
    script = tmp_path / 'synthetic.tcl'
    script.write_text('# synthetic fixture only\n')
    stages = ('floorplanning', 'placement', 'cts', 'routing')
    experiment = {'design': 'synthetic-fixture', 'initialization_script': str(script),
                  'extraction_script': str(script), 'stage_scripts': {s: str(script) for s in stages},
                  'stage_actions': {s: [{'name': 'baseline'}, {'name': 'alternate'}] for s in stages},
                  'reward_scales': {'area_um2': 1}, 'reward_directions': {'area_um2': -1}}
    expfile = tmp_path / 'experiment.json'
    expfile.write_text(json.dumps(experiment))
    cfg = _read_config(Path('configs/training.yaml'))
    cfg.update(openroad_executable=str(tool), openroad_experiment=str(expfile), rollout_workers=2,
               rollout_steps=4, updates=1, hidden_dim=8, device='cpu', ppo_epochs=2,
               wandb_enabled=True, checkpoint_episodes=2)
    run = train(cfg, tmp_path / 'run', synthetic=False)
    events = [json.loads(x) for x in (run / 'events.jsonl').read_text().splitlines()]
    assert sum(event['kind'] == 'stage' for event in events) == 8
    assert {event['stage'] for event in events if event['kind'] == 'stage_update'} == set(stages)
    assert (run / 'checkpoint-00001.pt').exists()
    assert len(list((run / 'workers' / 'update-00001').glob('worker-*'))) == 2
    cfg.update(updates=2, resume_checkpoint=str(run / 'checkpoint-00001.pt'))
    train(cfg, run, synthetic=False)
    assert (run / 'checkpoint-00002.pt').exists()
    assert ('start', None) in calls and ('start', 'test-run-id') in calls
    assert sum(c[0] == 'finish' for c in calls) == 2
    logged = next(c[2] for c in calls if c[0] == 'log')
    assert logged['training/episodes'] == 2
    assert 'training/floorplanning/actor_loss' in logged
    assert 'training/floorplanning/approx_kl' in logged
    assert 'training/mean_episode_reward' in logged
