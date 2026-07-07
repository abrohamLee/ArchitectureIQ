from architectureiq.datasets import DatasetSpec, generate
from architectureiq.trainer import train_curve


def test_curve_length_and_monotone_ish():
    X, y, in_dim, n_classes = generate(
        DatasetSpec(family="modular_addition", n_samples=200, modulus=7), seed=0
    )
    curve = train_curve("mlp", X, y, in_dim, n_classes, steps=50, seed=0, eval_every=10)
    # 记录点:step 0,10,20,30,40,50 -> 6 个
    assert curve.steps == [0, 10, 20, 30, 40, 50]
    assert len(curve.loss) == 6 and len(curve.acc) == 6
    # 最终 loss 应低于初始 loss(至少在学)
    assert curve.loss[-1] < curve.loss[0]


def test_training_is_deterministic():
    X, y, in_dim, n_classes = generate(
        DatasetSpec(family="parity", n_samples=128, n_bits=8), seed=0
    )
    c1 = train_curve("mlp", X, y, in_dim, n_classes, steps=30, seed=1)
    c2 = train_curve("mlp", X, y, in_dim, n_classes, steps=30, seed=1)
    assert c1.loss == c2.loss and c1.acc == c2.acc
