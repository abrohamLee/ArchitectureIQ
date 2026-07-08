from architectureiq.doctor import doctor_config
from architectureiq.doctorstate import init_doctor, load_doctor, save_doctor


def test_roundtrip_preserves_budget(tmp_path):
    run_dir = str(tmp_path / "d")
    ep = init_doctor(run_dir, doctor_config("too_low"))
    ep.treat(1e-2)
    save_doctor(run_dir, ep)
    loaded = load_doctor(run_dir)
    assert loaded.budget_spent == ep.budget_spent
    assert loaded.config.sick_lr == ep.config.sick_lr


def test_load_missing_raises(tmp_path):
    try:
        load_doctor(str(tmp_path / "nope"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
