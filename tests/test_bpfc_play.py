import json

from architectureiq.bpforecast import baseline_miss, load_hard_real
from architectureiq.play import main


def _run(capsys, argv):
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_hard_bank_only_keeps_baseline_failures():
    hard = load_hard_real("data/real_curves")
    assert len(hard) > 50
    assert all(baseline_miss(i) >= 0.15 for i in hard)


def test_bpfc_play_end_to_end(tmp_path, capsys):
    run_dir = str(tmp_path / "f")
    code, obs = _run(capsys, ["bpfc-init", "--run-dir", run_dir, "--index", "0"])
    assert code == 0 and "horizon_step" in obs and obs["reveal_cap_step"] < obs["horizon_step"]
    _run(capsys, ["bpfc-reveal", "--run-dir", run_dir, "--until", str(obs["reveal_cap_step"])])
    _, res = _run(capsys, ["bpfc-predict", "--run-dir", run_dir, "--value", str(obs["prefix_values"][-1])])
    assert "skill" in res and "baseline_pred" in res and "truth" in res
