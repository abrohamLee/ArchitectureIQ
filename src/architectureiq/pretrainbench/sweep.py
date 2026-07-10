"""V0 两决策轴 sweep(spec §2/§9):lr×warmup 12 点 + data mix 4 点,去重后 15 点 × seeds。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RunConfig, config_hash

BASE_MIX = {"web": 0.6, "code": 0.3, "math": 0.1}
LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3)
WARMUP_GRID = (0.0, 0.02, 0.1)
MIX_GRID = (
    {"web": 0.8, "code": 0.1, "math": 0.1},
    {"web": 0.6, "code": 0.3, "math": 0.1},  # 基准点,与 lr 网格 (3e-4, 0.02) 重合,去重时归并
    {"web": 0.4, "code": 0.3, "math": 0.3},
    {"web": 0.2, "code": 0.5, "math": 0.3},
)


def v0_sweep(scale: str, seeds: int = 3) -> list[RunConfig]:
    cfgs: list[RunConfig] = []
    seen: set[tuple[str, int]] = set()
    for seed in range(seeds):
        for lr in LR_GRID:
            for wu in WARMUP_GRID:
                cfgs.append(RunConfig(scale=scale, peak_lr=lr, warmup_frac=wu,
                                      decay="cosine", data_mix=dict(BASE_MIX), seed=seed))
        for mix in MIX_GRID:
            cfgs.append(RunConfig(scale=scale, peak_lr=3e-4, warmup_frac=0.02,
                                  decay="cosine", data_mix=dict(mix), seed=seed))
    out = []
    for c in cfgs:
        key = (config_hash(c), c.seed)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


SBATCH = """#!/bin/bash
#SBATCH --job-name=pretrainbench-v0
#SBATCH --array=0-{last}
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {out}/manifest.jsonl)
echo "$CFG" > /tmp/cfg_$SLURM_ARRAY_TASK_ID.json
uv run python -m architectureiq.pretrainbench.trainer \\
  --config /tmp/cfg_$SLURM_ARRAY_TASK_ID.json --shards {shards} --out {runlib}
"""


def write_sweep(out_dir: Path, scale: str, seeds: int, shard_dir: str, runlib_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfgs = v0_sweep(scale, seeds)
    with (out / "manifest.jsonl").open("w") as f:
        for c in cfgs:
            f.write(json.dumps(c.to_dict()) + "\n")
    (out / "sweep.sbatch").write_text(
        SBATCH.format(last=len(cfgs) - 1, out=out, shards=shard_dir, runlib=runlib_dir))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="130m")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--runlib", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(write_sweep(Path(a.out), a.scale, a.seeds, a.shards, a.runlib))


if __name__ == "__main__":
    main()
