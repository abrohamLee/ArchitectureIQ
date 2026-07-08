from architectureiq.baselines import run_random_probe_blackbox
from architectureiq.blackbox import default_blackbox_config


def test_random_probe_baseline_runs_and_spends_one_probe():
    cfg = default_blackbox_config("mlp")
    res = run_random_probe_blackbox(cfg)
    assert res.chosen in cfg.pool
    assert res.steps_spent > 0
