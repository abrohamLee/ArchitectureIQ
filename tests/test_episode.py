from architectureiq.datasets import DatasetSpec
from architectureiq.episode import Environment, default_config


def test_probe_deducts_budget_and_returns_margin():
    env = Environment(default_config())
    start = env.config.budget_steps
    res = env.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert res.cost == env.probe_cost()
    assert res.budget_remaining == start - res.cost
    assert not res.over_budget
    assert res.margin > 0.0


def test_probe_refuses_when_over_budget_without_charging():
    cfg = default_config()
    env = Environment(cfg, budget_spent=cfg.budget_steps)  # 预算已耗尽
    res = env.probe(DatasetSpec(family="random", n_samples=300, modulus=7))
    assert res.over_budget is True
    assert res.budget_remaining == 0


def test_commit_structured_dataset_scores_positive():
    env = Environment(default_config())
    res = env.commit(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert res.correct is True
    assert res.score > 0.0


def test_commit_generic_dataset_scores_zero():
    env = Environment(default_config())
    res = env.commit(DatasetSpec(family="random", n_samples=300, modulus=7))
    assert res.correct is False
    assert res.score == 0.0


def test_efficient_commit_beats_wasteful_commit():
    # 少花预算 -> 更高分
    lean = Environment(default_config(), budget_spent=0)
    waste = Environment(default_config(), budget_spent=6000)
    good = DatasetSpec(family="modular_addition", n_samples=300, modulus=7)
    assert lean.commit(good).score > waste.commit(good).score
