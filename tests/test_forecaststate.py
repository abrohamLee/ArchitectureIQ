from architectureiq.forecast import default_forecast_config
from architectureiq.forecaststate import init_forecast, load_forecast, save_forecast


def test_roundtrip_preserves_progress(tmp_path):
    run_dir = str(tmp_path / "f")
    ep = init_forecast(run_dir, default_forecast_config())
    ep.predict(0.9)
    save_forecast(run_dir, ep)
    loaded = load_forecast(run_dir)
    assert loaded.cursor == ep.cursor
    assert loaded.rounds == ep.rounds
    assert loaded.observed()["steps"] == ep.observed()["steps"]


def test_load_missing_raises(tmp_path):
    try:
        load_forecast(str(tmp_path / "nope"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
