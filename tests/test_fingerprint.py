from architectureiq.datasets import DatasetSpec, generate
from architectureiq.fingerprint import score_fingerprint, structure_advantage

ARCHS = ["mlp", "tiny_transformer"]


def test_structure_advantage_high_on_structured_low_on_generic():
    # 直接检验签名机制:结构数据上 MLP 的 Δ 明显 > 随机数据上的 Δ
    Xs, ys, ind, nc = generate(DatasetSpec(family="modular_addition", n_samples=300, modulus=7), seed=0)
    Xr, yr, _, _ = generate(DatasetSpec(family="random", n_samples=300, modulus=7), seed=0)
    delta_struct = structure_advantage("mlp", Xs, ys, ind, nc, steps=80, seed=2)
    delta_generic = structure_advantage("mlp", Xr, yr, ind, nc, steps=80, seed=2)
    # 用最终 checkpoint 的 Δ 比较
    assert delta_struct[-1] > 0.5
    assert abs(delta_generic[-1]) < 0.2


def test_separable_dataset_scores_high():
    # modular addition:两架构的 Δ-签名高置信可区分(accuracy + margin 都高)
    spec = DatasetSpec(family="modular_addition", n_samples=300, modulus=7)
    result = score_fingerprint(spec, ARCHS, ref_seeds=[0, 1], query_seeds=[2, 3], steps=80)
    assert result.accuracy >= 0.75
    assert result.margin >= 0.5


def test_generic_random_dataset_margin_collapses():
    # 随机标签:所有架构 Δ ≈ 0,签名塌到零。硬 NN-accuracy 在 2 架构小池上过脆
    # (残留噪声仍能凑到 ~0.75),但分离「置信度」margin 崩塌 —— 这才是失败信号。
    spec = DatasetSpec(family="random", n_samples=300, modulus=7)
    result = score_fingerprint(spec, ARCHS, ref_seeds=[0, 1], query_seeds=[2, 3], steps=80)
    assert result.margin <= 0.35


def test_score_is_deterministic():
    spec = DatasetSpec(family="modular_addition", n_samples=200, modulus=7)
    r1 = score_fingerprint(spec, ARCHS, [0], [1], steps=40)
    r2 = score_fingerprint(spec, ARCHS, [0], [1], steps=40)
    assert r1 == r2
