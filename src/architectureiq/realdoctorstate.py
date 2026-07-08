import json
import os

from architectureiq.realdoctor import RealDoctorConfig, RealDoctorEpisode

_STATE = "dr_real_state.json"


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, _STATE)


def init_real_doctor(run_dir: str, curve_id: str) -> RealDoctorEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = RealDoctorEpisode(RealDoctorConfig(curve_id=curve_id))
    save_real_doctor(run_dir, ep)
    return ep


def save_real_doctor(run_dir: str, ep: RealDoctorEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    c = ep.config
    payload = {
        "config": {
            "curve_id": c.curve_id, "init_frac": c.init_frac,
            "reveal_frac": c.reveal_frac, "human_budget": c.human_budget,
            "threshold": c.threshold,
        },
        "cursor": ep.cursor, "budget_spent": ep.budget_spent,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_real_doctor(run_dir: str) -> RealDoctorEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    config = RealDoctorConfig(**p["config"])
    return RealDoctorEpisode(config, cursor=p["cursor"], budget_spent=p["budget_spent"])
