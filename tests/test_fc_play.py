import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_fc_init_reports_observed(tmp_path, capsys):
    run_dir = str(tmp_path / "f")
    code, obs = _run(capsys, ["fc-init", "--run-dir", run_dir])
    assert code == 0
    assert obs["steps"] == [0, 20]
    assert obs["next_step"] == 40
    assert obs["done"] is False


def test_fc_predict_returns_skill_and_advances(tmp_path, capsys):
    run_dir = str(tmp_path / "f")
    main(["fc-init", "--run-dir", run_dir]); capsys.readouterr()
    code, res = _run(capsys, ["fc-predict", "--run-dir", run_dir, "--value", "0.95"])
    assert code == 0
    assert "skill" in res and "truth" in res
    _, obs = _run(capsys, ["fc-observe", "--run-dir", run_dir])
    assert obs["next_step"] == 60
