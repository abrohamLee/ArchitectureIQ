from architectureiq.baselines import run_successive_halving
from architectureiq.tournament import Tournament, default_tournament_config


def _prior_informed_pick(cfg):
    """脚本化 prior:各探一眼早期曲线(便宜),选早期最优者 answer。

    利用「早期 acc 预测最终赢家」这个语义先验;regret 基于真实 final_acc,
    故无需把赢家训到底 —— 纯 best-arm identification,花最少预算。
    """
    t = Tournament(cfg)
    for c in cfg.candidates:
        t.advance(c.id, cfg.eval_every)  # 每个候选各看一小段早期曲线
    best_early = max(cfg.candidates, key=lambda c: t.advance(c.id, 0).acc)
    return t.answer(best_early.id)


def test_prior_informed_beats_or_matches_sh_on_efficiency():
    cfg = default_tournament_config()
    agent = _prior_informed_pick(cfg)
    sh = run_successive_halving(cfg, eta=2)
    # prior-informed 应找到合格候选(correct),且不比语义盲的 SH 花更多步
    assert agent.correct is True
    assert agent.agent_steps <= sh.steps_spent
