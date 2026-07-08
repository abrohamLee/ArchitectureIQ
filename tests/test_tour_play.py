import json

from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_tour_init_reports_candidates_and_budget(tmp_path, capsys):
    run_dir = str(tmp_path / "t")
    code, obs = _run(capsys, ["tour-init", "--run-dir", run_dir])
    assert code == 0
    assert obs["budget_spent"] == 0
    assert len(obs["candidates"]) == 8
    assert obs["max_steps"] > 0


def test_tour_advance_then_observe_reflects_trained(tmp_path, capsys):
    run_dir = str(tmp_path / "t")
    main(["tour-init", "--run-dir", run_dir]); capsys.readouterr()
    cid = "mlp_0.01"
    code, res = _run(capsys, ["tour-advance", "--run-dir", run_dir, "--candidate", cid, "--steps", "40"])
    assert code == 0 and res["trained_steps"] == 40 and res["cost"] == 40
    _, obs = _run(capsys, ["tour-observe", "--run-dir", run_dir])
    assert obs["trained"][cid] == 40 and obs["budget_spent"] == 40


def test_tour_answer_best_scores_positive(tmp_path, capsys):
    run_dir = str(tmp_path / "t")
    main(["tour-init", "--run-dir", run_dir]); capsys.readouterr()
    code, res = _run(capsys, ["tour-answer", "--run-dir", run_dir, "--candidate", "mlp_0.01"])
    assert code == 0
    assert res["correct"] is True and res["score"] > 0.0
