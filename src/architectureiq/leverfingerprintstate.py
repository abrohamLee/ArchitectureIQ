import json
import os

from architectureiq.leverfingerprint import LeverFingerprintEpisode, LeverFPConfig


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, "fpl_state.json")


def init_leverfp(run_dir: str, config: LeverFPConfig) -> LeverFingerprintEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = LeverFingerprintEpisode(config)
    save_leverfp(run_dir, ep)
    return ep


def save_leverfp(run_dir: str, ep: LeverFingerprintEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    payload = {"config": ep.config.__dict__, "budget_spent": ep.budget_spent}
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_leverfp(run_dir: str) -> LeverFingerprintEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    return LeverFingerprintEpisode(LeverFPConfig(**p["config"]), budget_spent=p["budget_spent"])
