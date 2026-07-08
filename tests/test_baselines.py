from architectureiq.baselines import random_spec, run_random_agent
from architectureiq.episode import default_config


def test_random_spec_is_deterministic_and_valid():
    s1 = random_spec(0)
    s2 = random_spec(0)
    assert s1 == s2
    assert s1.family in ("modular_addition", "parity", "random")


def test_random_agent_produces_a_commit_result():
    res = run_random_agent(default_config(), n_probes=3, rng_seed=0)
    assert 0.0 <= res.score <= 1.0
    assert isinstance(res.correct, bool)
