import json

from architectureiq.play import main


def _obs(capsys):
    return json.loads(capsys.readouterr().out)


def test_prior_cli_play_scores_positive_and_is_lean(tmp_path, capsys):
    run_dir = str(tmp_path / "t")
    main(["tour-init", "--run-dir", run_dir]); capsys.readouterr()
    # 读早期信号:各推进一小段(eval_every),挑早期 acc 最高者 answer
    main(["tour-observe", "--run-dir", run_dir]); obs = _obs(capsys)
    best_id, best_acc = None, -1.0
    for cid in obs["candidates"]:
        main(["tour-advance", "--run-dir", run_dir, "--candidate", cid, "--steps", str(obs["eval_every"])])
        acc = _obs(capsys)["acc"]
        if acc > best_acc:
            best_acc, best_id = acc, cid
    main(["tour-answer", "--run-dir", run_dir, "--candidate", best_id])
    res = _obs(capsys)
    assert res["correct"] is True
    # 只各探一小段就 answer:总花费 = 8 × eval_every,远小于把全部训到 max
    assert res["agent_steps"] <= len(obs["candidates"]) * obs["eval_every"]


def test_readme_documents_all_four_actions():
    with open("tasks/tournament/README.md") as f:
        text = f.read()
    for cmd in ("tour-init", "tour-observe", "tour-advance", "tour-answer"):
        assert cmd in text
