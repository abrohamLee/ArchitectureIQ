"""单 run 训练器:全量 observable 免费记录(spec §2)。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import DOMAINS, RunConfig, config_hash
from .data import MixLoader
from .model import GPT

SPIKE_FACTOR = 1.5  # loss > 1.5x 近期中位数 => spike 事件


def _lr_at(cfg: RunConfig, step: int) -> float:
    total, warm = cfg.preset.total_steps, max(1, int(cfg.warmup_frac * cfg.preset.total_steps))
    if step < warm:
        return cfg.peak_lr * (step + 1) / warm
    frac = (step - warm) / max(1, total - warm)
    if cfg.decay == "cosine":
        return cfg.peak_lr * 0.5 * (1 + math.cos(math.pi * frac))
    if cfg.decay == "linear":
        return cfg.peak_lr * (1 - frac)
    return cfg.peak_lr


@torch.no_grad()
def _eval_ce(model: GPT, shard_dir: Path, cfg: RunConfig, n_batches: int = 4) -> dict:
    model.eval()
    out = {}
    for dom in DOMAINS:
        pure = RunConfig(**{**cfg.to_dict(), "data_mix": {d: 1.0 if d == dom else 0.0 for d in DOMAINS}})
        it = MixLoader(shard_dir, pure, split="val").batches()
        ces = []
        for _ in range(n_batches):
            x, y = next(it)
            logits = model(x)
            ces.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
        out[f"val_ce_{dom}"] = sum(ces) / len(ces)
    out["val_ce"] = sum(cfg.data_mix[d] * out[f"val_ce_{d}"] for d in DOMAINS)
    model.train()
    return out


def train_run(cfg: RunConfig, shard_dir: Path, out_dir: Path,
              log_every: int = 1, eval_every: int = 10) -> Path:
    run_dir = Path(out_dir) / f"{config_hash(cfg)}-s{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))

    torch.manual_seed(cfg.seed)
    model = GPT(cfg.preset)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr, weight_decay=0.1)
    batches = MixLoader(shard_dir, cfg, split="train").batches()

    recent, n_spikes = [], 0
    with (run_dir / "log.jsonl").open("w") as flog, (run_dir / "eval.jsonl").open("w") as fev:
        for step in range(cfg.preset.total_steps):
            lr = _lr_at(cfg, step)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = next(batches)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            opt.step()

            lv = loss.item()
            med = sorted(recent)[len(recent) // 2] if recent else lv
            spike = bool(recent) and lv > SPIKE_FACTOR * med
            n_spikes += spike
            recent = (recent + [lv])[-20:]

            if step % log_every == 0:
                wn = sum(p.detach().norm() ** 2 for p in model.parameters()).sqrt().item()
                flog.write(json.dumps({"step": step, "loss": lv, "lr": lr,
                                       "grad_norm": grad_norm, "weight_norm": wn,
                                       "spike": spike}) + "\n")
            if step % eval_every == 0 or step == cfg.preset.total_steps - 1:
                fev.write(json.dumps({"step": step, **_eval_ce(model, shard_dir, cfg)}) + "\n")

    final = json.loads((run_dir / "eval.jsonl").read_text().splitlines()[-1])
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_dir.name, "config_hash": config_hash(cfg), "seed": cfg.seed,
        "status": "done", "final_val_ce": final["val_ce"], "n_spikes": n_spikes}, indent=1))
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = RunConfig.from_dict(json.loads(Path(a.config).read_text()))
    print(train_run(cfg, Path(a.shards), Path(a.out)))


if __name__ == "__main__":
    main()
