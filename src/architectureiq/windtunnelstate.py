import json
import os

from architectureiq.windtunnel import WindTunnel, WindTunnelConfig


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, "wt_state.json")


def init_windtunnel(run_dir: str, config: WindTunnelConfig) -> WindTunnel:
    os.makedirs(run_dir, exist_ok=True)
    wt = WindTunnel(config)
    save_windtunnel(run_dir, wt)
    return wt


def save_windtunnel(run_dir: str, wt: WindTunnel) -> None:
    os.makedirs(run_dir, exist_ok=True)
    payload = {"config": wt.config.__dict__, "budget_spent": wt.budget_spent}
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f, indent=2)


def load_windtunnel(run_dir: str) -> WindTunnel:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    return WindTunnel(WindTunnelConfig(**p["config"]), budget_spent=p["budget_spent"])
