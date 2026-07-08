from architectureiq.forecast import ForecastEpisode, linear_extrapolate
from architectureiq.realcurvebank import filter_ids, get_curve, list_curve_ids

REAL = "pythia|pythia-1.4b|zero-shot|piqa"


def test_bank_has_many_diverse_curves():
    ids = list_curve_ids()
    assert len(ids) >= 100  # 大幅扩充:数百条真实曲线
    # 多样性:多个模型尺寸、多个任务、多种 shot
    models = {i.split("|")[1] for i in ids}
    tasks = {i.split("|")[3] for i in ids}
    shots = {i.split("|")[2] for i in ids}
    assert len(models) >= 8 and len(tasks) >= 5 and len(shots) >= 2


def test_real_curve_loads_with_expected_shape():
    assert REAL in list_curve_ids()
    curve = get_curve(REAL)
    assert len(curve.acc) == len(curve.steps) >= 20
    assert 0.5 <= min(curve.acc) < max(curve.acc) <= 0.75
    assert curve.steps[0] == 0 and curve.steps[-1] == 143000


def test_filter_ids_by_dimension():
    piqa_ids = filter_ids(task="piqa")
    assert all(i.endswith("|piqa") for i in piqa_ids)
    assert len(piqa_ids) >= 8  # 多个模型的 piqa 曲线


def test_forecast_episode_runs_on_injected_real_curve():
    ep = ForecastEpisode(curve=get_curve(REAL))
    rounds = 0
    while not ep.is_done():
        obs = ep.observed()
        ep.predict(obs["values"][-1])  # persistence
        rounds += 1
    assert rounds >= 1 and isinstance(ep.score(), float)


def test_persistence_beats_linear_on_real_curve():
    # 真实 LLM 曲线(噪声 + 亚-1 饱和)上,persistence 打败线性外推(skill>0)。
    # 玩具版管用的「clamp 到 1」在此无效 —— real tier 测的动力学直觉不同。
    curve = get_curve(REAL)
    ep = ForecastEpisode(curve=curve)
    while not ep.is_done():
        ep.predict(ep.observed()["values"][-1])
    assert ep.score() > 0.0

    ep2 = ForecastEpisode(curve=curve)
    while not ep2.is_done():
        obs = ep2.observed()
        ep2.predict(linear_extrapolate(obs["steps"], obs["values"], ep2.next_step()))
    assert abs(ep2.score()) < 1e-9  # 线性外推自己当预测 -> skill 恒 0
