"""Synthetic telemetry contract checks; no W&B network traffic."""
import sys
from types import SimpleNamespace

import pytest

from mappo_eda.telemetry import physical_metrics, public_config, start_wandb


def test_wandb_resume_uses_run_id_and_never_forwards_credentials(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: calls.append(kw)))
    config = {'seed': 7, 'resume_checkpoint': 'checkpoint.pt', 'wandb_api_key': 'synthetic-secret'}
    start_wandb(config, tmp_path, 'run-123')
    assert calls[0]['id'] == 'run-123' and calls[0]['resume'] == 'must'
    assert calls[0]['project'] == 'MAPPO-EDA' and calls[0]['mode'] == 'online'
    assert 'synthetic-secret' not in repr(calls)
    assert public_config({'nested': {'api_key': 'synthetic-secret', 'seed': 7}}) == {'nested': {'seed': 7}}
    with pytest.raises(ValueError, match='no W&B run ID'):
        start_wandb(config, tmp_path)


def test_only_real_supplied_values_and_defined_baselines_are_logged():
    result = physical_metrics({'area_um2': 8, 'wns_ns': -0.2}, {'area_um2': 10, 'wns_ns': 0}, {'area_um2': -1, 'wns_ns': 1})
    assert result['improvement/area_um2_percent'] == 20
    assert 'improvement/wns_ns_percent' not in result
    assert 'physical/power' not in result
