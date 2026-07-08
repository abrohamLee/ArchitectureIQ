import pytest

from architectureiq.harbor import score_submission


def test_blackbox_lever_correctness():
    hidden = {"lever_family": "optimizer", "hidden_value": "sgd"}
    assert score_submission("blackbox_lever", hidden, {"guess": "sgd"}) == 1.0
    assert score_submission("blackbox_lever", hidden, {"guess": "adam"}) == 0.0


def test_diagnostic_correctness():
    hidden = {"pathology": "vanishing_grad"}
    assert score_submission("diagnostic", hidden, {"cause": "vanishing_grad"}) == 1.0
    assert score_submission("diagnostic", hidden, {"cause": "dead_relu"}) == 0.0


def test_windtunnel_rewards_true_best_only():
    from architectureiq.windtunnel import WindTunnel, default_windtunnel_config
    seed = 3
    hidden = {"seed": seed, "regret_threshold": 0.05}
    wt = WindTunnel(default_windtunnel_config(seed))
    best = wt.best
    worst = min(wt.large, key=wt.large.get)
    assert score_submission("windtunnel", hidden, {"candidate": best}) == 1.0
    assert score_submission("windtunnel", hidden, {"candidate": worst}) == 0.0
    assert score_submission("windtunnel", hidden, {"candidate": "ZZ"}) == 0.0


def test_fingerprint_lever_structured_vs_random():
    hidden = {
        "lever_family": "optimizer", "correct_margin": 0.22,
        "ref_seeds": [10, 11, 12], "query_seeds": [13, 14, 15], "probe_steps": 80,
    }
    good = score_submission("fingerprint_lever", hidden,
                            {"family": "modular_addition", "n_samples": 300, "modulus": 7})
    bad = score_submission("fingerprint_lever", hidden,
                           {"family": "random", "n_samples": 300, "modulus": 7})
    assert good == 1.0 and bad == 0.0


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        score_submission("nope", {}, {})
