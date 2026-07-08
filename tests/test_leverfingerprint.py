from architectureiq.datasets import DatasetSpec
from architectureiq.leverfingerprint import (
    LeverFingerprintEpisode,
    default_leverfp_config,
    score_lever_fingerprint,
)

STRUCT = DatasetSpec(family="modular_addition", n_samples=300, modulus=7)
RANDOM = DatasetSpec(family="random", n_samples=300, modulus=7)


def test_structured_separates_levers_random_collapses():
    for family in ["optimizer", "activation"]:
        s = score_lever_fingerprint(family, STRUCT, [0, 1, 2], [3, 4, 5], steps=80)
        r = score_lever_fingerprint(family, RANDOM, [0, 1, 2], [3, 4, 5], steps=80)
        assert s.margin >= 0.22          # 结构数据分开杠杆
        assert r.margin < 0.22           # random 塌陷
        assert s.margin > r.margin * 2   # 明显差距


def test_probe_deducts_budget():
    ep = LeverFingerprintEpisode(default_leverfp_config("optimizer"))
    start = ep.config.budget_steps
    res = ep.probe(STRUCT)
    assert res.cost == ep.probe_cost() and res.budget_remaining == start - res.cost
    assert res.margin > 0.0


def test_commit_structured_correct_random_zero():
    good = LeverFingerprintEpisode(default_leverfp_config("optimizer")).commit(STRUCT)
    assert good.correct is True and good.score > 0.0
    bad = LeverFingerprintEpisode(default_leverfp_config("optimizer")).commit(RANDOM)
    assert bad.correct is False and bad.score == 0.0
