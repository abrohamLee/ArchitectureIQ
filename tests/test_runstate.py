from architectureiq.episode import Environment, default_config
from architectureiq.runstate import init_state, load_state, save_state


def test_roundtrip_preserves_budget_and_config(tmp_path):
    run_dir = str(tmp_path / "run1")
    env = init_state(run_dir, default_config())
    env.budget_spent = 2400
    env.committed = True
    save_state(run_dir, env)

    loaded = load_state(run_dir)
    assert loaded.budget_spent == 2400
    assert loaded.committed is True
    assert loaded.config.archs == env.config.archs
    assert loaded.config.correct_margin == env.config.correct_margin


def test_load_missing_raises(tmp_path):
    try:
        load_state(str(tmp_path / "nope"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
