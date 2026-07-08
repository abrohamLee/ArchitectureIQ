from architectureiq.forecast import (
    ForecastEpisode,
    default_forecast_config,
    linear_extrapolate,
)


def _saturation_aware_play(cfg):
    ep = ForecastEpisode(cfg)
    while not ep.is_done():
        obs = ep.observed()
        raw = linear_extrapolate(obs["steps"], obs["values"], ep.next_step())
        pred = min(1.0, max(0.0, raw))  # 懂 acc 饱和在 [0,1]
        ep.predict(pred)
    return ep.score()


def test_saturation_aware_beats_linear_baseline():
    score = _saturation_aware_play(default_forecast_config())
    # 平均 skill > 0 即整体打败裸线性外推
    assert score > 0.0


def test_readme_documents_actions():
    with open("tasks/forecast/README.md") as f:
        text = f.read()
    for cmd in ("fc-init", "fc-observe", "fc-predict"):
        assert cmd in text
