import numpy as np
from architectureiq.pretrainbench.config import RunConfig
from architectureiq.pretrainbench.data import MixLoader, make_test_shards


def cfg(mix=None, seed=0):
    return RunConfig(scale="test", peak_lr=3e-4, warmup_frac=0.05, decay="cosine",
                     data_mix=mix or {"web": 0.6, "code": 0.3, "math": 0.1}, seed=seed)


def test_shapes_and_shift(tmp_path):
    make_test_shards(tmp_path, vocab=512, tokens_per_domain=50_000)
    loader = MixLoader(tmp_path, cfg(), split="train")
    x, y = next(loader.batches())
    assert x.shape == (8, 64) and y.shape == (8, 64)
    assert (x[:, 1:] == y[:, :-1]).all()  # y 是 x 右移一位


def test_deterministic_given_seed(tmp_path):
    make_test_shards(tmp_path, vocab=512, tokens_per_domain=50_000)
    a = next(MixLoader(tmp_path, cfg(seed=3), split="train").batches())[0]
    b = next(MixLoader(tmp_path, cfg(seed=3), split="train").batches())[0]
    c = next(MixLoader(tmp_path, cfg(seed=4), split="train").batches())[0]
    assert (a == b).all() and not (a == c).all()


def test_mix_weights_respected(tmp_path):
    # 用可区分的 token 区间标记域:web=[0,100) code=[100,200) math=[200,300)
    make_test_shards(tmp_path, vocab=512, tokens_per_domain=50_000)
    loader = MixLoader(tmp_path, cfg(mix={"web": 1.0, "code": 0.0, "math": 0.0}), split="train")
    x, _ = next(loader.batches())
    assert int(x.max()) < 100  # 纯 web 配比只出 web 区间 token
