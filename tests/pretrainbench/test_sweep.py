import json
from architectureiq.pretrainbench.config import config_hash
from architectureiq.pretrainbench.sweep import v0_sweep, write_sweep


def test_sweep_size_and_uniqueness():
    cfgs = v0_sweep("test", seeds=3)
    # lr×warmup 12 点 + mix 4 点,其中 mix 网格的基准配比点与 lr 网格 (3e-4, 0.02) 重合,去重后 15 点
    assert len(cfgs) == 15 * 3
    assert len({(config_hash(c), c.seed) for c in cfgs}) == len(cfgs)  # 无重复点
    assert len({config_hash(c) for c in cfgs}) == 15                   # 15 个唯一实验点


def test_write_sweep(tmp_path):
    out = write_sweep(tmp_path, "test", seeds=2, shard_dir="/shards", runlib_dir="/runlib")
    lines = (out / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 30 and json.loads(lines[0])["scale"] == "test"
    sbatch = (out / "sweep.sbatch").read_text()
    assert "#SBATCH --array=0-29" in sbatch and "architectureiq.pretrainbench.trainer" in sbatch
