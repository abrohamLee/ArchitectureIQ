import torch
from architectureiq.pretrainbench.config import SCALES
from architectureiq.pretrainbench.model import GPT


def test_gpt_forward_shape_and_causality():
    m = GPT(SCALES["test"])
    x = torch.randint(0, 512, (2, 64))
    logits = m(x)
    assert logits.shape == (2, 64, 512)
    # 因果性:改动位置 j 的输入不影响位置 < j 的输出
    x2 = x.clone()
    x2[:, 40] = (x2[:, 40] + 1) % 512
    with torch.no_grad():
        a, b = m(x), m(x2)
    assert torch.allclose(a[:, :40], b[:, :40], atol=1e-5)
    assert not torch.allclose(a[:, 40:], b[:, 40:], atol=1e-5)


def test_param_count_scale():
    # test 档 ~170k 参数(2 层 d64 词表 512),量级正确即可
    assert 100_000 < GPT(SCALES["test"]).num_params() < 5_000_000
