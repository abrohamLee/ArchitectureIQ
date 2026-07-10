import pytest
from architectureiq.pretrainbench.config import RunConfig, SCALES, config_hash


def make_cfg(**over):
    base = dict(scale="test", peak_lr=3e-4, warmup_frac=0.05, decay="cosine",
                data_mix={"web": 0.6, "code": 0.3, "math": 0.1}, seed=0)
    base.update(over)
    return RunConfig(**base)


def test_roundtrip():
    cfg = make_cfg()
    assert RunConfig.from_dict(cfg.to_dict()) == cfg


def test_hash_ignores_seed_but_not_axes():
    cfg = make_cfg()
    assert config_hash(cfg) == config_hash(make_cfg(seed=7))          # seed 不进哈希
    assert config_hash(cfg) != config_hash(make_cfg(peak_lr=1e-3))    # 决策轴进哈希
    assert config_hash(cfg) != config_hash(make_cfg(data_mix={"web": 1.0, "code": 0.0, "math": 0.0}))
    assert len(config_hash(cfg)) == 12


def test_mix_must_sum_to_one():
    with pytest.raises(ValueError):
        make_cfg(data_mix={"web": 0.9, "code": 0.3, "math": 0.1})


def test_scale_presets():
    assert SCALES["test"].n_layer == 2 and SCALES["test"].total_steps == 30
    assert SCALES["130m"].n_embd == 768 and SCALES["130m"].n_layer == 12
