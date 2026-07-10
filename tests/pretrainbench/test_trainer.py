import json

import torch
from architectureiq.pretrainbench.config import SCALES, RunConfig
from architectureiq.pretrainbench.data import make_test_shards
from architectureiq.pretrainbench.model import GPT
from architectureiq.pretrainbench.trainer import train_run


def test_gpt_forward_shape_and_causality():
    m = GPT(SCALES["test"])
    x = torch.randint(0, 512, (2, 64))
    logits = m(x)
    assert logits.shape == (2, 64, 512)
    # 因果性:改动位置 j 的输入不影响位置 < j 的输出
    x2 = x.clone()
    x2[:, 40] = (x2[:, 40] + 1) % 512
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :40], b[:, :40], atol=1e-5)
    assert not torch.allclose(a[:, 40:], b[:, 40:], atol=1e-5)


def test_param_count_scale():
    # test 档 ~170k 参数(2 层 d64 词表 512),量级正确即可
    assert 100_000 < GPT(SCALES["test"]).num_params() < 5_000_000


def _cfg(seed=0, peak_lr=3e-3):
    return RunConfig(scale="test", peak_lr=peak_lr, warmup_frac=0.1, decay="cosine",
                     data_mix={"web": 0.6, "code": 0.3, "math": 0.1}, seed=seed)


def test_train_run_outputs(tmp_path):
    make_test_shards(tmp_path / "shards", vocab=512, tokens_per_domain=50_000)
    run_dir = train_run(_cfg(), tmp_path / "shards", tmp_path / "runlib")
    logs = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    evals = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines()]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert len(logs) == 30 and {"step", "loss", "lr", "grad_norm", "weight_norm", "spike"} <= logs[0].keys()
    assert evals and {"step", "val_ce", "val_ce_web", "val_ce_code", "val_ce_math"} <= evals[0].keys()
    assert meta["status"] == "done" and run_dir.name == f"{meta['config_hash']}-s0"
    # warmup 生效:第 1 步 lr < 第 3 步 lr(warmup 3 步)
    assert logs[0]["lr"] < logs[2]["lr"]
    # 训练有效:loss 下降
    assert logs[-1]["loss"] < logs[0]["loss"]


def test_reproducible_same_seed(tmp_path):
    make_test_shards(tmp_path / "shards", vocab=512, tokens_per_domain=50_000)
    d1 = train_run(_cfg(seed=1), tmp_path / "shards", tmp_path / "a")
    d2 = train_run(_cfg(seed=1), tmp_path / "shards", tmp_path / "b")
    assert (d1 / "log.jsonl").read_text() == (d2 / "log.jsonl").read_text()
