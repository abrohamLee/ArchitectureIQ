import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_bbl_init_hides_answer(tmp_path, capsys):
    run_dir = str(tmp_path / "l")
    code, obs = _run(capsys, ["bbl-init", "--run-dir", run_dir, "--family", "optimizer", "--hidden", "rmsprop"])
    assert code == 0
    assert "hidden_value" not in obs and "hidden" not in obs  # 不泄露答案
    assert set(obs["pool"]) == {"adam", "sgd", "rmsprop"}


def test_bbl_probe_then_guess(tmp_path, capsys):
    run_dir = str(tmp_path / "l")
    main(["bbl-init", "--run-dir", run_dir, "--family", "optimizer", "--hidden", "sgd"]); capsys.readouterr()
    code, res = _run(capsys, ["bbl-probe", "--run-dir", run_dir, "--family", "modular_addition", "--n-samples", "300"])
    assert code == 0 and "sgd" in res["references"]
    _, g = _run(capsys, ["bbl-guess", "--run-dir", run_dir, "--value", "sgd"])
    assert g["correct"] is True and g["score"] > 0.0
