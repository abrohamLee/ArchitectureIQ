from architectureiq.baselines import run_successive_halving
from architectureiq.tournament import default_tournament_config


def test_sh_returns_one_survivor_within_budget():
    cfg = default_tournament_config()
    res = run_successive_halving(cfg, eta=2)
    assert res.chosen_id in {c.id for c in cfg.candidates}
    assert res.steps_spent > 0
    assert res.regret >= 0.0


def test_sh_regret_is_bounded():
    # SH 是合理地板:regret 不应大到离谱(应找到接近最优的候选)
    cfg = default_tournament_config()
    res = run_successive_halving(cfg, eta=2)
    assert res.regret <= 0.5
