import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_fpl_init_reports_lever_and_budget(tmp_path, capsys):
    run_dir = str(tmp_path / "f")
    code, obs = _run(capsys, ["fpl-init", "--run-dir", run_dir, "--lever", "activation"])
    assert code == 0
    assert obs["lever_family"] == "activation"
    assert set(obs["lever_values"]) == {"relu", "tanh", "gelu"}
    assert obs["probe_cost"] > 0 and obs["budget_spent"] == 0


def test_fpl_commit_structured_scores_positive(tmp_path, capsys):
    run_dir = str(tmp_path / "f")
    main(["fpl-init", "--run-dir", run_dir, "--lever", "optimizer"]); capsys.readouterr()
    code, res = _run(capsys, ["fpl-commit", "--run-dir", run_dir, "--family", "modular_addition", "--n-samples", "300"])
    assert code == 0
    assert res["correct"] is True and res["score"] > 0.0
