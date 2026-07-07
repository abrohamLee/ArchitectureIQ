from architectureiq.discriminator import curve_features, discriminate
from architectureiq.trainer import LearningCurve


def _curve(loss, acc):
    return LearningCurve(steps=list(range(len(loss))), loss=loss, acc=acc)


def test_curve_features_concatenates_loss_and_acc():
    feats = curve_features(_curve([1.0, 0.5], [0.1, 0.9]))
    assert feats == [1.0, 0.5, 0.1, 0.9]


def test_discriminate_perfect_when_archs_separable():
    ref = {
        "mlp": [_curve([2.0, 2.0], [0.1, 0.1])],
        "tiny_transformer": [_curve([2.0, 0.0], [0.1, 1.0])],
    }
    queries = [
        ("mlp", _curve([2.0, 1.9], [0.1, 0.15])),
        ("tiny_transformer", _curve([2.0, 0.1], [0.1, 0.95])),
    ]
    acc, margin = discriminate(ref, queries)
    assert acc == 1.0
    assert margin > 0.0


def test_discriminate_chance_when_archs_identical():
    # 两个 arch 的曲线完全重叠 -> 无法区分,准确率不应高于瞎猜
    same = [_curve([1.0, 1.0], [0.5, 0.5])]
    ref = {"mlp": same, "tiny_transformer": same}
    queries = [
        ("mlp", _curve([1.0, 1.0], [0.5, 0.5])),
        ("tiny_transformer", _curve([1.0, 1.0], [0.5, 0.5])),
    ]
    acc, margin = discriminate(ref, queries)
    assert acc <= 0.5
    assert margin == 0.0
