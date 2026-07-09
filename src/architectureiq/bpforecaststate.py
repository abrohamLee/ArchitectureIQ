import json
import os

from architectureiq.bpforecast import BPForecastEpisode, FCInstance


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, "bpfc_state.json")


def init_bpforecast(run_dir: str, instance: FCInstance) -> BPForecastEpisode:
    os.makedirs(run_dir, exist_ok=True)
    ep = BPForecastEpisode(instance)
    save_bpforecast(run_dir, ep)
    return ep


def save_bpforecast(run_dir: str, ep: BPForecastEpisode) -> None:
    os.makedirs(run_dir, exist_ok=True)
    i = ep.inst
    payload = {
        "instance": {"id": i.id, "config": i.config, "metric": i.metric,
                     "steps": i.steps, "values": i.values,
                     "free_frac": i.free_frac, "reveal_cap_frac": i.reveal_cap_frac},
        "revealed_until_step": ep.revealed_until_step,
    }
    with open(_path(run_dir), "w") as f:
        json.dump(payload, f)


def load_bpforecast(run_dir: str) -> BPForecastEpisode:
    path = _path(run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        p = json.load(f)
    d = p["instance"]
    inst = FCInstance(id=d["id"], config=d["config"], metric=d["metric"],
                      steps=d["steps"], values=d["values"],
                      free_frac=d["free_frac"], reveal_cap_frac=d["reveal_cap_frac"])
    return BPForecastEpisode(inst, revealed_until_step=p["revealed_until_step"])
