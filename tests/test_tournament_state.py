from architectureiq.tournament import Tournament, default_tournament_config


def test_snapshot_reflects_advances():
    t = Tournament(default_tournament_config())
    cid = t.config.candidates[0].id
    t.advance(cid, 60)
    snap = t.snapshot()
    assert snap["trained"][cid] == 60
    assert snap["budget_spent"] == 60


def test_restored_tournament_resumes_budget_and_trained():
    cfg = default_tournament_config()
    cid = cfg.candidates[0].id
    restored = Tournament(cfg, trained={c.id: (60 if c.id == cid else 0) for c in cfg.candidates}, budget_spent=60)
    # 从 60 继续推进 40 -> 100,增量 40
    r = restored.advance(cid, 40)
    assert r.trained_steps == 100 and r.cost == 40
    assert restored.budget_spent == 100
