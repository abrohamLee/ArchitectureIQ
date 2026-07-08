from architectureiq.blackbox import (
    BlackboxEpisode,
    default_blackbox_config,
    nearest,
)
from architectureiq.datasets import DatasetSpec


def test_probe_returns_refs_for_pool_and_mystery():
    ep = BlackboxEpisode(default_blackbox_config("mlp"))
    res = ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert set(res.references.keys()) == set(ep.config.pool)
    assert len(res.mystery) == len(next(iter(res.references.values())))
    assert res.cost == ep.probe_cost()
    assert res.budget_remaining == ep.config.budget_steps - res.cost


def test_structured_probe_identifies_hidden_via_nearest():
    ep = BlackboxEpisode(default_blackbox_config("mlp"))
    res = ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert nearest(res.mystery, res.references) == "mlp"  # 好探测认得出


def test_guess_correct_scores_positive_wrong_zero():
    ep = BlackboxEpisode(default_blackbox_config("gru"))
    ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    good = ep.guess("gru")
    assert good.correct is True and good.score > 0.0
    ep2 = BlackboxEpisode(default_blackbox_config("gru"))
    bad = ep2.guess("mlp")
    assert bad.correct is False and bad.score == 0.0


def test_probe_over_budget_does_not_charge():
    cfg = default_blackbox_config("mlp")
    ep = BlackboxEpisode(cfg, budget_spent=cfg.budget_steps)
    res = ep.probe(DatasetSpec(family="random", n_samples=300, modulus=7))
    assert res.over_budget is True
