from architectureiq.windtunnel import WindTunnel, default_windtunnel_config

K = len(default_windtunnel_config(0).scale_costs)


def _trust_small(seed):
    wt = WindTunnel(default_windtunnel_config(seed))
    v = {i: wt.run(i, 0).value for i in wt.ids}
    return wt.commit(max(v, key=v.get))


def _scale_ladder(seed):
    wt = WindTunnel(default_windtunnel_config(seed))
    v0 = {i: wt.run(i, 0).value for i in wt.ids}
    t3 = sorted(v0, key=v0.get, reverse=True)[:3]
    v2 = {i: wt.run(i, 2).value for i in t3}
    t2 = sorted(v2, key=v2.get, reverse=True)[:2]
    vB = {i: wt.run(i, K - 1).value for i in t2}
    return wt.commit(max(vB, key=vB.get))


def test_build_emergence_small_compressed_large_spread():
    assert len(WindTunnel(default_windtunnel_config(0)).ids) == 6
    # 涌现是平均性质:小尺度候选挤在一起(方差小),大尺度才拉开
    def spread(xs):
        return max(xs) - min(xs)
    small_avg = large_avg = 0.0
    for s in range(20):
        wt = WindTunnel(default_windtunnel_config(s))
        small_avg += spread([wt.value_at(i, 0) for i in wt.ids])
        large_avg += spread([wt.value_at(i, K - 1) for i in wt.ids])
    assert large_avg > small_avg  # 平均上大尺度更拉开


def test_run_deducts_and_over_budget_guard():
    wt = WindTunnel(default_windtunnel_config(0))
    r = wt.run("A", 2)
    assert r.cost == wt.config.scale_costs[2] and not r.over_budget
    full = default_windtunnel_config(0).budget
    wt2 = WindTunnel(default_windtunnel_config(0), budget_spent=full)
    assert wt2.run("A", K - 1).over_budget is True


def test_commit_best_scores_positive():
    wt = WindTunnel(default_windtunnel_config(3))
    res = wt.commit(wt.best)
    assert res.correct is True and res.score > 0.0


def test_scale_ladder_beats_trusting_small_scale():
    # 核心 headline:爬到大尺度验证的策略,正确率远高于信便宜代理
    seeds = range(40)
    ladder_correct = sum(_scale_ladder(s).correct for s in seeds)
    trust_correct = sum(_trust_small(s).correct for s in seeds)
    assert ladder_correct > trust_correct + 8  # 25 vs 9,留足余量
    # 且平均分也更高
    ladder_score = sum(_scale_ladder(s).score for s in seeds) / 40
    trust_score = sum(_trust_small(s).score for s in seeds) / 40
    assert ladder_score > trust_score
