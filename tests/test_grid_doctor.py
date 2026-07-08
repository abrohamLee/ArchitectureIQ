from architectureiq.baselines import run_grid_search_doctor
from architectureiq.doctor import doctor_config


def test_grid_search_finds_cure_but_spends_full_grid():
    cfg = doctor_config("too_high")
    res = run_grid_search_doctor(cfg)
    assert res.correct is True
    # 试遍 5 个 lr,每个 max_steps
    assert res.steps_spent == len(cfg.grid) * cfg.max_steps
