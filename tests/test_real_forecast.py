from architectureiq.forecast import ForecastEpisode, linear_extrapolate
from architectureiq.realcurvebank import list_real_curves, load_real_curve

REAL = "pythia-1.4b_piqa_zeroshot"


def test_real_curve_loads_with_expected_shape():
    assert REAL in list_real_curves()
    curve = load_real_curve(REAL)
    assert len(curve.acc) == len(curve.steps) >= 20
    # piqa acc:从瞎猜(~0.5)升到 ~0.71
    assert 0.5 <= min(curve.acc) < max(curve.acc) <= 0.75
    assert curve.steps[0] == 0 and curve.steps[-1] == 143000


def test_forecast_episode_runs_on_injected_real_curve():
    curve = load_real_curve(REAL)
    ep = ForecastEpisode(curve=curve)
    rounds = 0
    while not ep.is_done():
        obs = ep.observed()
        ep.predict(obs["values"][-1])  # persistence
        rounds += 1
    assert rounds >= 1
    assert isinstance(ep.score(), float)


def test_persistence_beats_linear_on_real_curve():
    # 真实 LLM 曲线(噪声 + 亚-1 饱和)上,persistence 打败线性外推基线(skill>0)。
    # 注意:玩具版管用的「clamp 到 1」在此无效(acc 够不到 1)—— real tier 测的动力学直觉不同。
    curve = load_real_curve(REAL)
    ep = ForecastEpisode(curve=curve)
    while not ep.is_done():
        obs = ep.observed()
        ep.predict(obs["values"][-1])
    assert ep.score() > 0.0

    # 对照:纯线性外推自己当预测 -> skill 恒 0(它就是基线)
    ep2 = ForecastEpisode(curve=curve)
    while not ep2.is_done():
        obs = ep2.observed()
        raw = linear_extrapolate(obs["steps"], obs["values"], ep2.next_step())
        ep2.predict(raw)
    assert abs(ep2.score()) < 1e-9
