from architectureiq.tournament import default_tournament_config
from architectureiq.tourstate import init_tournament, load_tournament, save_tournament


def test_roundtrip_preserves_state(tmp_path):
    run_dir = str(tmp_path / "t")
    t = init_tournament(run_dir, default_tournament_config())
    cid = t.config.candidates[0].id
    t.advance(cid, 80)
    save_tournament(run_dir, t)

    loaded = load_tournament(run_dir)
    assert loaded.budget_spent == 80
    assert loaded.snapshot()["trained"][cid] == 80
    assert [c.id for c in loaded.config.candidates] == [c.id for c in t.config.candidates]
    assert loaded.config.spec == t.config.spec


def test_load_missing_raises(tmp_path):
    try:
        load_tournament(str(tmp_path / "nope"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
