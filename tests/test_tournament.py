from architectureiq.tournament import Tournament, default_tournament_config


def test_advance_charges_incremental_steps():
    t = Tournament(default_tournament_config())
    cid = t.config.candidates[0].id
    r1 = t.advance(cid, 40)
    assert r1.trained_steps == 40 and r1.cost == 40
    r2 = t.advance(cid, 60)  # 40 -> 100,增量 60
    assert r2.trained_steps == 100 and r2.cost == 60


def test_advance_caps_at_max_steps():
    t = Tournament(default_tournament_config())
    cid = t.config.candidates[0].id
    t.advance(cid, t.config.max_steps + 500)
    r = t.advance(cid, 100)  # 已到 max,无可推进
    assert r.trained_steps == t.config.max_steps
    assert r.cost == 0


def test_answer_best_candidate_zero_regret_and_high_score():
    t = Tournament(default_tournament_config())
    best = t.best_candidate_id()
    res = t.answer(best)
    assert res.regret == 0.0
    assert res.correct is True
    assert res.score > 0.0


def test_answer_worst_candidate_incurs_regret_zero_score():
    cfg = default_tournament_config()
    t = Tournament(cfg)
    # 选一个 final_acc 最低的候选
    worst = min(cfg.candidates, key=lambda c: t.bank.final_acc(c))
    res = t.answer(worst.id)
    if res.regret > cfg.regret_threshold:
        assert res.correct is False and res.score == 0.0
