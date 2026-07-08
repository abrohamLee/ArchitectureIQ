import torch

from architectureiq.models import build_model


def test_mlp_output_shape_and_size():
    m = build_model("mlp", in_dim=14, n_classes=7)
    out = m(torch.randn(5, 14))
    assert out.shape == (5, 7)
    assert sum(p.numel() for p in m.parameters()) < 100_000


def test_tiny_transformer_output_shape_and_size():
    m = build_model("tiny_transformer", in_dim=14, n_classes=7)
    out = m(torch.randn(5, 14))
    assert out.shape == (5, 7)
    assert sum(p.numel() for p in m.parameters()) < 100_000


def test_gru_output_shape_and_size():
    m = build_model("gru", in_dim=14, n_classes=7)
    out = m(torch.randn(5, 14))
    assert out.shape == (5, 7)
    assert sum(p.numel() for p in m.parameters()) < 100_000


def test_cnn1d_output_shape_and_size():
    m = build_model("cnn1d", in_dim=14, n_classes=7)
    out = m(torch.randn(5, 14))
    assert out.shape == (5, 7)
    assert sum(p.numel() for p in m.parameters()) < 100_000


def test_unknown_arch_raises():
    try:
        build_model("lstm", in_dim=8, n_classes=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
