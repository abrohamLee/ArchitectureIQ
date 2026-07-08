import json
import re

from architectureiq.play import main

CURVE = "pythia|pythia-1.4b|zero-shot|piqa"


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def _size(cid: str) -> float:
    m = re.search(r"(\d+\.?\d*)([mb])", cid.split("|")[1])
    return float(m.group(1)) * (1000 if m.group(2) == "b" else 1)


def test_fc_real_init_then_predict(tmp_path, capsys):
    run_dir = str(tmp_path / "fr")
    code, obs = _run(capsys, ["fc-real-init", "--run-dir", run_dir, "--curve", CURVE])
    assert code == 0
    assert obs["done"] is False and obs["next_step"] is not None
    # 复用 fc-predict:real 状态 load 后能揭真打分
    _, res = _run(capsys, ["fc-predict", "--run-dir", run_dir, "--value", str(obs["values"][-1])])
    assert "skill" in res and "truth" in res
    # 复用 fc-observe:确认从 real 状态重建并前进
    _, obs2 = _run(capsys, ["fc-observe", "--run-dir", run_dir])
    assert len(obs2["values"]) == len(obs["values"]) + 1


def test_tour_real_init_then_answer_biggest(tmp_path, capsys):
    run_dir = str(tmp_path / "tr")
    code, obs = _run(capsys, ["tour-real-init", "--run-dir", run_dir, "--task", "piqa", "--shot", "zero-shot"])
    assert code == 0
    assert len(obs["candidates"]) >= 10
    # 凭 scale prior 直接 answer 最大模型(复用 tour-answer,real 状态 load)
    biggest = max(obs["candidates"], key=_size)
    _, res = _run(capsys, ["tour-answer", "--run-dir", run_dir, "--candidate", biggest])
    assert res["correct"] is True and res["score"] > 0.0
