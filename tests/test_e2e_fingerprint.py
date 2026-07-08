import json

from architectureiq.baselines import run_random_agent
from architectureiq.episode import default_config
from architectureiq.play import main


def test_smart_commit_beats_random_agent(tmp_path, capsys):
    # 聪明 agent:凭 prior 直接一次 commit modular_addition(0 探针)
    run_dir = str(tmp_path / "smart")
    main(["init", "--run-dir", run_dir])
    capsys.readouterr()
    main(["commit", "--run-dir", run_dir, "--family", "modular_addition", "--n-samples", "300"])
    smart = json.loads(capsys.readouterr().out)

    # 随机 agent:瞎试 6 次再 commit
    rand = run_random_agent(default_config(), n_probes=6, rng_seed=1)

    assert smart["correct"] is True
    assert smart["score"] > rand.score


def test_readme_documents_all_four_actions():
    with open("tasks/fingerprint/README.md") as f:
        text = f.read()
    for cmd in ("init", "observe", "probe", "commit"):
        assert cmd in text
