import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_bb_init_and_observe_hide_arch(tmp_path, capsys):
    run_dir = str(tmp_path / "b")
    code, obs = _run(capsys, ["bb-init", "--run-dir", run_dir, "--hidden", "gru"])
    assert code == 0
    # 池列出所有候选,但不能有字段泄露「哪个是黑盒」
    assert "hidden_arch" not in obs and "hidden" not in obs
    assert set(obs["pool"]) == {"mlp", "tiny_transformer", "gru"}


def test_bb_probe_then_guess(tmp_path, capsys):
    run_dir = str(tmp_path / "b")
    main(["bb-init", "--run-dir", run_dir, "--hidden", "mlp"]); capsys.readouterr()
    code, res = _run(capsys, ["bb-probe", "--run-dir", run_dir, "--family", "modular_addition", "--n-samples", "300"])
    assert code == 0 and "mlp" in res["references"]
    _, g = _run(capsys, ["bb-guess", "--run-dir", run_dir, "--arch", "mlp"])
    assert g["correct"] is True and g["score"] > 0.0
