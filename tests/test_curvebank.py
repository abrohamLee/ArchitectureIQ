from architectureiq.curvebank import Candidate, CurveBank
from architectureiq.datasets import DatasetSpec


def _bank():
    spec = DatasetSpec(family="modular_addition", n_samples=300, modulus=7)
    return CurveBank(spec, max_steps=60, eval_every=20)


def test_acc_at_is_prefix_of_full_curve():
    bank = _bank()
    cand = Candidate(id="mlp_1e-2", arch="mlp", lr=1e-2)
    # checkpoints: 0,20,40,60
    assert bank.checkpoints() == [0, 20, 40, 60]
    # 训到 40 步的 acc = 完整曲线在 checkpoint 40 的值
    full = bank.curve(cand)
    idx40 = full.steps.index(40)
    assert bank.acc_at(cand, 45) == full.acc[idx40]  # 45 向下取到 40
    assert bank.final_acc(cand) == full.acc[-1]


def test_bank_caches_and_is_deterministic():
    bank = _bank()
    cand = Candidate(id="mlp_1e-2", arch="mlp", lr=1e-2)
    c1 = bank.curve(cand)
    c2 = bank.curve(cand)  # 缓存命中,同一对象/同值
    assert c1.acc == c2.acc


def test_mlp_beats_random_lr_choice_on_modular_addition():
    # 好 lr 的 MLP 最终 acc 应显著高于极小 lr 的(候选间有真实排序)
    bank = _bank()
    good = bank.final_acc(Candidate("mlp_1e-2", "mlp", 1e-2))
    bad = bank.final_acc(Candidate("mlp_1e-4", "mlp", 1e-4))
    assert good > bad
