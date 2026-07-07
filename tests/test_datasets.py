import torch

from architectureiq.datasets import DatasetSpec, generate


def test_modular_addition_shapes_and_labels():
    spec = DatasetSpec(family="modular_addition", n_samples=50, modulus=7)
    X, y, in_dim, n_classes = generate(spec, seed=0)
    assert X.shape == (50, 14)
    assert y.shape == (50,)
    assert in_dim == 14 and n_classes == 7
    assert X.dtype == torch.float32 and y.dtype == torch.long
    # 每行恰好两个 1(a 的 one-hot + b 的 one-hot)
    assert torch.all(X.sum(dim=1) == 2)
    # 标签在 [0, 7)
    assert int(y.min()) >= 0 and int(y.max()) < 7


def test_parity_labels_are_bit_sum_mod_2():
    spec = DatasetSpec(family="parity", n_samples=64, n_bits=8)
    X, y, in_dim, n_classes = generate(spec, seed=0)
    assert in_dim == 8 and n_classes == 2
    expected = (X.sum(dim=1).long() % 2)
    assert torch.equal(y, expected)


def test_generate_is_deterministic():
    spec = DatasetSpec(family="modular_addition", n_samples=30)
    X1, y1, _, _ = generate(spec, seed=3)
    X2, y2, _, _ = generate(spec, seed=3)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)


def test_random_family_labels_uniform_not_structured():
    spec = DatasetSpec(family="random", n_samples=200, modulus=7)
    _, y, _, n_classes = generate(spec, seed=0)
    assert n_classes == 7
    # 至少用到多个类(不是常数标签)
    assert len(torch.unique(y)) >= 5
