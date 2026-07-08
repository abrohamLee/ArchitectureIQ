from architectureiq.forecast import linear_extrapolate, skill_score


def test_linear_extrapolate_continues_slope():
    # (0,0.1),(20,0.5) -> 斜率 0.02/step -> 在 40 应为 0.9
    assert abs(linear_extrapolate([0, 20], [0.1, 0.5], 40) - 0.9) < 1e-9


def test_linear_extrapolate_single_point_returns_last():
    assert linear_extrapolate([0], [0.3], 20) == 0.3


def test_skill_positive_when_agent_beats_baseline():
    # agent 误差 0.1,baseline 误差 0.5 -> skill 0.8
    assert abs(skill_score(0.1, 0.5) - 0.8) < 1e-9


def test_skill_negative_when_worse_than_baseline():
    assert skill_score(1.0, 0.5) < 0.0


def test_skill_zero_baseline_perfect():
    assert skill_score(0.0, 0.0) == 1.0
    assert skill_score(0.2, 0.0) == 0.0
