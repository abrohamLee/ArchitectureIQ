from architectureiq.diagnostic import (
    PATHOLOGIES,
    DiagnosticEpisode,
    default_diagnostic_config,
)


def _data(p):
    return DiagnosticEpisode(default_diagnostic_config(p))._data


def test_raw_loss_indistinguishable_across_pathologies():
    # litmus 前半:三病因的 loss 曲线终点都卡在高位平台,彼此难分
    finals = [_data(p)["loss_curve"][-1] for p in PATHOLOGIES]
    assert all(f > 1.5 for f in finals)          # 都是"高位平台"(病态)
    assert max(finals) - min(finals) < 0.25      # 彼此聚在一起 → raw 曲线分不出


def test_observables_separate_the_pathologies():
    # litmus 后半:在不同 observable 上可分
    d = {p: _data(p) for p in PATHOLOGIES}
    assert d["dead_relu"]["dead_fraction"] > 0.6        # 死 ReLU:高死亡率
    assert d["vanishing_grad"]["dead_fraction"] < 0.2   # 消失:死亡率低
    # 逐层梯度早/晚比:消失 & 死ReLU 低、lr太低 正常
    def ratio(p):
        g = d[p]["per_layer_grad"]; return (g[0] + 1e-9) / (g[-1] + 1e-9)
    assert ratio("lr_too_low") > 0.15
    assert ratio("vanishing_grad") < 0.15


def test_query_deducts_budget_and_over_budget():
    ep = DiagnosticEpisode(default_diagnostic_config("dead_relu"))
    r = ep.query("dead_fraction")
    assert r.cost == 1 and not r.over_budget and r.value > 0.6
    ep2 = DiagnosticEpisode(default_diagnostic_config("dead_relu"), budget_spent=10)
    assert ep2.query("per_layer_grad").over_budget is True


def _diagnose(pathology):
    """脚本化诊断者:查 dead_fraction + per_layer_grad,按阈值定因(2 次 query)。"""
    ep = DiagnosticEpisode(default_diagnostic_config(pathology))
    dead = ep.query("dead_fraction").value
    pl = ep.query("per_layer_grad").value
    ratio = (pl[0] + 1e-9) / (pl[-1] + 1e-9)
    if dead > 0.6:
        cause = "dead_relu"
    elif ratio < 0.15:
        cause = "vanishing_grad"
    else:
        cause = "lr_too_low"
    return ep.answer(cause)


def test_query_diagnoser_beats_no_query_baseline():
    # 查对 observable 的诊断者:三病因全对、正分
    for p in PATHOLOGIES:
        res = _diagnose(p)
        assert res.correct is True and res.score > 0.0
    # 非智能地板:不查、总猜多数病因 → 只对 1/3
    correct = sum(
        DiagnosticEpisode(default_diagnostic_config(p)).answer("lr_too_low").correct
        for p in PATHOLOGIES
    )
    assert correct == 1  # 3 个里只蒙对 1 个
