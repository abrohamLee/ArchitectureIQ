import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_wt_init_lists_candidates_and_scales(tmp_path, capsys):
    run_dir = str(tmp_path / "w")
    code, obs = _run(capsys, ["wt-init", "--run-dir", run_dir, "--seed", "3"])
    assert code == 0
    assert len(obs["candidates"]) == 6
    assert obs["scales"][0]["cost"] < obs["scales"][-1]["cost"]  # 大尺度更贵
    assert "large" not in json.dumps(obs)  # 不泄露大尺度真值


def test_wt_run_then_commit(tmp_path, capsys):
    run_dir = str(tmp_path / "w")
    main(["wt-init", "--run-dir", run_dir, "--seed", "3"]); capsys.readouterr()
    code, r = _run(capsys, ["wt-run", "--run-dir", run_dir, "--candidate", "A", "--scale", "2"])
    assert code == 0 and r["cost"] > 0 and 0.0 <= r["value"] <= 1.0
    _, obs = _run(capsys, ["wt-observe", "--run-dir", run_dir])
    assert obs["budget_spent"] == r["cost"]
    _, c = _run(capsys, ["wt-commit", "--run-dir", run_dir, "--candidate", "A"])
    assert "score" in c and "regret" in c
