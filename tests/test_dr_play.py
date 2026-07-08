import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_dr_init_reports_sick_curve_and_grid(tmp_path, capsys):
    run_dir = str(tmp_path / "d")
    code, obs = _run(capsys, ["dr-init", "--run-dir", run_dir, "--pathology", "too_high"])
    assert code == 0
    assert "loss" in obs["sick_curve"]
    assert 0.01 in obs["grid"]
    assert obs["budget_spent"] == 0


def test_dr_treat_then_commit(tmp_path, capsys):
    run_dir = str(tmp_path / "d")
    main(["dr-init", "--run-dir", run_dir]); capsys.readouterr()
    code, tr = _run(capsys, ["dr-treat", "--run-dir", run_dir, "--lr", "0.01"])
    assert code == 0 and tr["cured"] is True
    _, cr = _run(capsys, ["dr-commit", "--run-dir", run_dir, "--lr", "0.01"])
    assert cr["correct"] is True and cr["score"] > 0.0
