"""端到端冒烟:test 档迷你 sweep(2 seed)→ registry → 闸门筛可成题配置对。
用法: uv run python scripts/pretrainbench_smoke.py /tmp/pbench_smoke
"""
import sys
from itertools import combinations
from pathlib import Path

from architectureiq.pretrainbench.data import make_test_shards
from architectureiq.pretrainbench.gate import pair_gate
from architectureiq.pretrainbench.registry import Registry
from architectureiq.pretrainbench.sweep import v0_sweep
from architectureiq.pretrainbench.trainer import train_run

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pbench_smoke")
make_test_shards(root / "shards", vocab=512, tokens_per_domain=50_000)
for cfg in v0_sweep("test", seeds=2):
    train_run(cfg, root / "shards", root / "runlib")

reg = Registry(root / "runlib")
groups = reg.groups()
print(f"runs={len(reg.runs())} points={len(groups)}")
valid = 0
for ha, hb in combinations(groups, 2):
    r = pair_gate(reg.final(ha), reg.final(hb))
    if r.passed:
        valid += 1
        print(f"OK  {ha} vs {hb}  gap={r.gap:.3f} win={r.win_rate:.2f} better={r.better}")
print(f"\n可成题配置对: {valid}/{len(groups) * (len(groups) - 1) // 2}")
