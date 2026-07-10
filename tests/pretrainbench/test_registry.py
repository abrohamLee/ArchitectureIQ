from architectureiq.pretrainbench.config import RunConfig
from architectureiq.pretrainbench.data import make_test_shards
from architectureiq.pretrainbench.registry import Registry
from architectureiq.pretrainbench.trainer import train_run


def test_registry_groups_and_curves(tmp_path):
    make_test_shards(tmp_path / "shards", vocab=512, tokens_per_domain=50_000)
    base = dict(scale="test", warmup_frac=0.1, decay="cosine",
                data_mix={"web": 0.6, "code": 0.3, "math": 0.1})
    for lr in (3e-3, 3e-4):
        for seed in (0, 1):
            train_run(RunConfig(peak_lr=lr, seed=seed, **base), tmp_path / "shards", tmp_path / "runlib")
    reg = Registry(tmp_path / "runlib")
    assert len(reg.runs()) == 4
    groups = reg.groups()
    assert len(groups) == 2 and all(len(v) == 2 for v in groups.values())
    some_id = reg.runs()[0]["run_id"]
    curve = reg.curve(some_id, "loss")
    assert len(curve) == 30 and curve[0][0] == 0
    ev = reg.curve(some_id, "val_ce_code")
    assert ev and isinstance(ev[0][1], float)
    h = next(iter(groups))
    assert len(reg.final(h)) == 2
