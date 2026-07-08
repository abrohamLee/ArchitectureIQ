import re

from architectureiq.baselines import run_successive_halving
from architectureiq.tournament import Tournament, real_tournament_config


def _size(cid: str) -> float:
    m = re.search(r"(\d+\.?\d*)([mb])", cid.split("|")[1])
    return float(m.group(1)) * (1000 if m.group(2) == "b" else 1)


def test_real_tournament_builds_from_pythia_curves():
    cfg, bank = real_tournament_config("piqa", "zero-shot")
    assert len(cfg.candidates) >= 10
    assert cfg.max_steps == 143000
    # 真最优 = 最大模型(scale 单调)
    best = Tournament(cfg, bank=bank).best_candidate_id()
    assert "12b" in best


def test_scale_prior_beats_successive_halving_on_real_curves():
    cfg, bank = real_tournament_config("piqa", "zero-shot")

    # scale-prior:懂 scale laws,直接挑最大模型,零 advance
    biggest = max(cfg.candidates, key=lambda c: _size(c.id))
    scale = Tournament(cfg, bank=bank).answer(biggest.id)
    assert scale.correct is True
    assert scale.regret == 0.0
    assert scale.agent_steps == 0
    assert scale.score == 1.0

    # 语义盲 SH 被真曲线的早期噪声骗:花大量步数,regret 反而更高
    sh = run_successive_halving(cfg, eta=2, bank=bank)
    assert sh.steps_spent > 0
    assert sh.regret > scale.regret  # SH 没找到真最优
