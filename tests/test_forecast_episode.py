from architectureiq.forecast import ForecastEpisode, default_forecast_config


def test_observed_grows_and_next_step_advances():
    ep = ForecastEpisode(default_forecast_config())
    obs0 = ep.observed()
    assert obs0["steps"] == [0, 20]  # cursor=1 -> 见 checkpoint 0,20
    assert ep.next_step() == 40
    ep.predict(0.9)
    assert ep.observed()["steps"] == [0, 20, 40]
    assert ep.next_step() == 60


def test_perfect_prediction_beats_baseline_on_saturating_curve():
    # 完美预测(误差 0)不劣于基线 -> skill 非负
    peek = ForecastEpisode(default_forecast_config())
    truth_next = peek.predict(0.0).truth  # 乱预测拿到 truth
    ep = ForecastEpisode(default_forecast_config())
    r2 = ep.predict(truth_next)  # 完美预测
    assert r2.agent_err == 0.0
    assert r2.skill >= 0.0


def test_rolls_to_done_and_scores():
    ep = ForecastEpisode(default_forecast_config())
    while not ep.is_done():
        obs = ep.observed()
        pred = obs["values"][-1]  # persistence
        ep.predict(pred)
    assert isinstance(ep.score(), float)
