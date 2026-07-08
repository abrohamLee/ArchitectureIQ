import math

from architectureiq.blackbox import nearest  # 复用最近邻
from architectureiq.datasets import DatasetSpec
from architectureiq.leverid import LeverIDEpisode, default_leverid_config


def _margin(res):
    ds = sorted(math.sqrt(sum((a - b) ** 2 for a, b in zip(res.mystery, s)))
                for s in res.references.values())
    return ds[1] - ds[0]


def test_probe_returns_refs_for_lever_pool():
    ep = LeverIDEpisode(default_leverid_config("optimizer", "sgd"))
    res = ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert set(res.references.keys()) == {"adam", "sgd", "rmsprop"}
    assert res.cost == ep.probe_cost() and not res.over_budget


def test_structured_probe_identifies_hidden_lever():
    ep = LeverIDEpisode(default_leverid_config("optimizer", "sgd"))
    res = ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert nearest(res.mystery, res.references) == "sgd"  # 好探测认得出真实优化器


def test_structured_probe_beats_random_margin():
    # 结构探测的鉴定 margin 应大于 random 探测(默认探针塌陷 → 必须设计)
    struct = LeverIDEpisode(default_leverid_config("optimizer", "sgd")).probe(
        DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    rand = LeverIDEpisode(default_leverid_config("optimizer", "sgd")).probe(
        DatasetSpec(family="random", n_samples=300, modulus=7))
    assert _margin(struct) > _margin(rand)


def test_guess_scoring():
    ep = LeverIDEpisode(default_leverid_config("optimizer", "rmsprop"))
    ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert ep.guess("rmsprop").correct is True and ep.guess("rmsprop").score > 0.0
    ep2 = LeverIDEpisode(default_leverid_config("optimizer", "rmsprop"))
    assert ep2.guess("adam").correct is False and ep2.guess("adam").score == 0.0
