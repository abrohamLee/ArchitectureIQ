import torch

from architectureiq.determinism import set_determinism


def test_same_seed_same_randn():
    set_determinism(0)
    a = torch.randn(4)
    set_determinism(0)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_different_seed_different_randn():
    set_determinism(0)
    a = torch.randn(4)
    set_determinism(1)
    b = torch.randn(4)
    assert not torch.equal(a, b)
