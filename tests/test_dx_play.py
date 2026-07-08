import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_dx_init_shows_curve_and_hides_answer(tmp_path, capsys):
    run_dir = str(tmp_path / "d")
    code, obs = _run(capsys, ["dx-init", "--run-dir", run_dir, "--pathology", "vanishing_grad"])
    assert code == 0
    assert "loss_curve" in obs["sick_curve"] or isinstance(obs["sick_curve"], list)
    assert set(obs["causes"]) == {"lr_too_low", "dead_relu", "vanishing_grad"}
    assert "pathology" not in obs and "vanishing_grad" not in json.dumps(obs["observables"])


def test_dx_query_then_answer(tmp_path, capsys):
    run_dir = str(tmp_path / "d")
    main(["dx-init", "--run-dir", run_dir, "--pathology", "dead_relu"]); capsys.readouterr()
    code, q = _run(capsys, ["dx-query", "--run-dir", run_dir, "--observable", "dead_fraction"])
    assert code == 0 and q["value"] > 0.6 and q["cost"] == 1
    _, a = _run(capsys, ["dx-answer", "--run-dir", run_dir, "--cause", "dead_relu"])
    assert a["correct"] is True and a["score"] > 0.0
