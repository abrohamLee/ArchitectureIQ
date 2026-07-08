from architectureiq.blackbox import default_blackbox_config
from architectureiq.blackboxstate import init_blackbox, load_blackbox, save_blackbox


def test_roundtrip_preserves_budget_and_hidden(tmp_path):
    run_dir = str(tmp_path / "b")
    ep = init_blackbox(run_dir, default_blackbox_config("gru"))
    ep.budget_spent = 1120
    save_blackbox(run_dir, ep)
    loaded = load_blackbox(run_dir)
    assert loaded.budget_spent == 1120
    assert loaded.config.hidden_arch == "gru"
    assert loaded.config.pool == ep.config.pool


def test_load_missing_raises(tmp_path):
    try:
        load_blackbox(str(tmp_path / "nope"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
