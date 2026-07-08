import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_init_then_observe_reports_budget(tmp_path, capsys):
    run_dir = str(tmp_path / "r")
    code, obs = _run(capsys, ["init", "--run-dir", run_dir])
    assert code == 0
    assert obs["budget_spent"] == 0
    assert obs["probe_cost"] > 0
    assert "mlp" in obs["archs"]


def test_probe_then_observe_reflects_spent(tmp_path, capsys):
    run_dir = str(tmp_path / "r")
    main(["init", "--run-dir", run_dir])
    capsys.readouterr()
    code, res = _run(capsys, ["probe", "--run-dir", run_dir, "--family", "modular_addition", "--n-samples", "300"])
    assert code == 0
    assert res["cost"] > 0 and res["margin"] > 0.0
    _, obs = _run(capsys, ["observe", "--run-dir", run_dir])
    assert obs["budget_spent"] == res["cost"]


def test_commit_structured_scores_positive(tmp_path, capsys):
    run_dir = str(tmp_path / "r")
    main(["init", "--run-dir", run_dir])
    capsys.readouterr()
    code, res = _run(capsys, ["commit", "--run-dir", run_dir, "--family", "modular_addition", "--n-samples", "300"])
    assert code == 0
    assert res["correct"] is True and res["score"] > 0.0
