import math

from architectureiq.bpforecast import (
    BPForecastEpisode,
    FCInstance,
    best_baseline,
    load_all_real,
    skill,
)


def _grok_curve():
    # 前 60% 在 chance(~0.1)平台,80% 处突刺到 ~1.0 —— 朴素外推必挂
    steps = list(range(0, 100))
    values = [0.1 + 0.001 * s for s in range(60)] + [0.1 + (s - 60) * 0.9 / 20 if s < 80 else 1.0 for s in range(60, 100)]
    return FCInstance(id="grok", config={"arch": "transformer", "task": "mod_add", "lr": 3e-3},
                      metric="acc", steps=steps, values=values)


def test_baseline_on_flat_prefix_predicts_flat():
    # 只看前 60%(平台),朴素基线外推到末步应 ~平台值,而非 grok 后的 1.0
    inst = _grok_curve()
    ps, pv = inst.steps[:60], inst.values[:60]
    _, pred = best_baseline(ps, pv, inst.steps[-1])
    assert pred < 0.4  # 基线预测远低于真值 1.0


def test_predict_grok_beats_baseline():
    inst = _grok_curve()
    ep = BPForecastEpisode(inst)
    ep.reveal(inst.steps[59])           # 揭示到平台末(拐点前)
    good = ep.predict(1.0)              # 押"会 grok"
    bad = BPForecastEpisode(inst); bad.reveal(inst.steps[59])
    flat = bad.predict(0.16)           # 押"继续平"
    assert good.skill > 0.8            # 押对 grok → 高 skill
    assert flat.skill < 0.1            # 押平 → 接近基线,skill 低


def test_cap_hides_transition_from_baseline():
    # reveal 封顶在 60%,grok 在 80% —— 揭示到封顶,基线仍看不到拐点、外推偏低
    inst = _grok_curve()
    ep = BPForecastEpisode(inst)
    ep.reveal(inst.steps[-1])          # 会被 cap 到 60%
    res = ep.predict(1.0)              # 押 grok
    assert res.skill > 0.5             # 拐点在封顶外,基线挂,押对 grok 仍赢
    assert res.baseline_pred < 0.6


def test_reveal_capped_before_horizon():
    inst = _grok_curve()
    ep = BPForecastEpisode(inst)
    r = ep.reveal(inst.steps[-1])
    assert r.at_cap and r.revealed_until <= inst.steps[int(len(inst.steps) * 0.6)]


def test_load_real_instances():
    insts = load_all_real("data/real_curves")
    assert len(insts) > 100
    s = insts[0]
    assert s.steps and s.values and s.config
