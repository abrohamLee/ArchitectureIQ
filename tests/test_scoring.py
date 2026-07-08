from architectureiq.scoring import brute_force_ratio, rhae_ml_score


def test_incorrect_scores_zero_regardless_of_efficiency():
    assert rhae_ml_score(correct=False, agent_steps=1, human_steps=1000) == 0.0


def test_correct_and_efficient_caps_at_one():
    # agent 比人类还省 -> min(1, h/a)=1 -> 1.0
    assert rhae_ml_score(correct=True, agent_steps=10, human_steps=1000) == 1.0


def test_squared_penalty_for_inefficiency():
    # 人类 10 步,agent 100 步 -> (10/100)^2 = 0.01
    assert abs(rhae_ml_score(correct=True, agent_steps=100, human_steps=10) - 0.01) < 1e-9


def test_zero_agent_steps_does_not_divide_by_zero():
    assert rhae_ml_score(correct=True, agent_steps=0, human_steps=5) == 1.0


def test_brute_force_ratio_below_one_when_agent_cheaper():
    assert brute_force_ratio(agent_steps=200, bruteforce_steps=1000) == 0.2
    assert brute_force_ratio(agent_steps=0, bruteforce_steps=0) == 0.0
