import math

from architectureiq.blackbox import BlackboxEpisode, default_blackbox_config, nearest
from architectureiq.datasets import DatasetSpec


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _id_margin(res):
    # 次近距离 - 最近距离(越大越无歧义)
    ds = sorted(_dist(res.mystery, s) for s in res.references.values())
    return ds[1] - ds[0]


def test_structured_probe_identifies_and_beats_random_margin():
    ep = BlackboxEpisode(default_blackbox_config("mlp"))
    struct = ep.probe(DatasetSpec(family="modular_addition", n_samples=300, modulus=7))
    assert nearest(struct.mystery, struct.references) == "mlp"  # 结构探测认得出

    ep2 = BlackboxEpisode(default_blackbox_config("mlp"))
    rand = ep2.probe(DatasetSpec(family="random", n_samples=300, modulus=7))
    # 结构探测的鉴定 margin 应大于 random 探测(好探测更无歧义)
    assert _id_margin(struct) > _id_margin(rand)


def test_readme_documents_actions():
    with open("tasks/blackbox/README.md") as f:
        text = f.read()
    for cmd in ("bb-init", "bb-observe", "bb-probe", "bb-guess"):
        assert cmd in text
