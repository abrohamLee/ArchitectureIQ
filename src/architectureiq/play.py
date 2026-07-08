import argparse
import json

from architectureiq.datasets import DatasetSpec
from architectureiq.episode import default_config
from architectureiq.runstate import init_state, load_state, save_state
from architectureiq.blackbox import default_blackbox_config
from architectureiq.blackboxstate import init_blackbox, load_blackbox, save_blackbox
from architectureiq.diagnostic import PATHOLOGIES, QUERY_COSTS, default_diagnostic_config
from architectureiq.diagnosticstate import (
    init_diagnostic,
    load_diagnostic,
    save_diagnostic,
)
from architectureiq.doctor import doctor_config
from architectureiq.doctorstate import init_doctor, load_doctor, save_doctor
from architectureiq.leverfingerprint import default_leverfp_config
from architectureiq.leverfingerprintstate import init_leverfp, load_leverfp, save_leverfp
from architectureiq.leverid import default_leverid_config
from architectureiq.leveridstate import init_leverid, load_leverid, save_leverid
from architectureiq.levers import lever_values
from architectureiq.windtunnel import default_windtunnel_config
from architectureiq.windtunnelstate import (
    init_windtunnel,
    load_windtunnel,
    save_windtunnel,
)
from architectureiq.realdoctorstate import (
    init_real_doctor,
    load_real_doctor,
    save_real_doctor,
)
from architectureiq.forecast import default_forecast_config
from architectureiq.forecaststate import (
    init_forecast,
    init_real_forecast,
    load_forecast,
    save_forecast,
)
from architectureiq.tournament import default_tournament_config
from architectureiq.tourstate import (
    init_real_tournament,
    init_tournament,
    load_tournament,
    save_tournament,
)

RULES = (
    "设计一个数据集(DatasetSpec),使 4 个架构在其上训练产生的「结构优势」"
    "签名彼此可区分。probe 花预算试探,commit 用 held-out seed 最终评分。"
    "分数 = 是否可区分(margin>=correct_margin) × 效率(越省预算越高,平方惩罚)。"
)
DSL_SCHEMA = {
    "family": ["modular_addition", "parity", "random"],
    "n_samples": "int",
    "modulus": "int (modular_addition/random)",
    "n_bits": "int (parity)",
    "label_noise": "float in [0,1]",
}


def spec_from_args(family, n_samples, modulus, n_bits, label_noise) -> DatasetSpec:
    return DatasetSpec(
        family=family, n_samples=n_samples, modulus=modulus,
        n_bits=n_bits, label_noise=label_noise,
    )


def _observe_dict(env) -> dict:
    return {
        "archs": env.config.archs,
        "budget_steps": env.config.budget_steps,
        "budget_spent": env.budget_spent,
        "probe_cost": env.probe_cost(),
        "correct_margin": env.config.correct_margin,
        "committed": env.committed,
        "rules": RULES,
        "dsl_schema": DSL_SCHEMA,
    }


TOUR_RULES = (
    "一组 candidate(arch×lr)在固定数据集上竞争。用 tour-advance 花预算把某候选"
    "多训几步、看它的部分 acc;用 tour-answer 挑出你认为最终最优的候选。分数 = "
    "regret 是否达标(选中接近真最优) × 效率(总 steps 越少越高,平方惩罚)。"
)


def _tour_observe_dict(t) -> dict:
    return {
        "candidates": [c.id for c in t.config.candidates],
        "budget_steps": t.config.budget_steps,
        "budget_spent": t.budget_spent,
        "max_steps": t.config.max_steps,
        "eval_every": t.config.eval_every,
        "trained": t.snapshot()["trained"],
        "regret_threshold": t.config.regret_threshold,
        "rules": TOUR_RULES,
    }


def _fc_observe_dict(ep) -> dict:
    obs = ep.observed()
    return {
        "steps": obs["steps"],
        "values": obs["values"],
        "next_step": None if ep.is_done() else ep.next_step(),
        "done": ep.is_done(),
        "score": ep.score(),
        "rules": "逐 checkpoint 揭示训练曲线的 acc。每步预测下一 checkpoint 的 acc;"
                 "评分 = 相对线性外推基线的技巧分(懂饱和/平台动力学者胜)。",
    }


DR_RULES = (
    "给你一条病态训练曲线(sick_curve)。读它的 loss 形状诊断病因:loss 曾冲高过起点="
    "LR 太高,应降;loss 平台不降=LR 太低,应升。用 dr-treat 试某个 lr,或凭诊断直接 "
    "dr-commit 开药。分数 = 是否治愈(final_acc≥cure_acc) × 效率(试药越少越高,平方惩罚)。"
)


DR_REAL_RULES = (
    "给你一条真实训练 loss 曲线,逐段揭示。判断这个 run 是否会**发散**(病态:loss 从最低点"
    "冲高)。dr-real-reveal 多揭一段(花预算),或凭前缀直接 dr-real-diagnose --label "
    "healthy|pathological。越早判对分越高(平方效率惩罚)。"
)


def _dr_real_observe_dict(ep) -> dict:
    o = ep.observe()  # 只给已揭前缀,不泄露曲线 id / 真值
    return {
        "steps": o["steps"], "loss": o["loss"],
        "frac_revealed": o["frac_revealed"], "budget_spent": o["budget_spent"],
        "labels": ["healthy", "pathological"], "rules": DR_REAL_RULES,
    }


def _dr_observe_dict(ep) -> dict:
    return {
        "sick_curve": ep.sick_curve(),
        "grid": ep.config.grid,
        "budget_steps": ep.config.budget_steps,
        "budget_spent": ep.budget_spent,
        "cure_acc": ep.config.cure_acc,
        "treat_cost": ep.treat_cost(),
        "rules": DR_RULES,
    }


BB_RULES = (
    "环境藏了池中一个架构。设计探测数据集(bb-probe),环境返回池中各架构的参考「结构优势」"
    "签名 + 黑盒的 mystery 签名;比对 mystery 最像哪个参考,用 bb-guess 猜。好探测(结构数据)"
    "让签名清晰可分,random 探测会让签名塌成一团难分。分数 = 猜对 × 效率(探测越少越高)。"
)


def _bb_observe_dict(ep) -> dict:
    return {
        "pool": ep.config.pool,
        "budget_steps": ep.config.budget_steps,
        "budget_spent": ep.budget_spent,
        "probe_cost": ep.probe_cost(),
        "rules": BB_RULES,
    }


FPL_RULES = (
    "设计一个数据集,让一个真实杠杆家族的答案(如优化器 {adam,sgd,rmsprop} / 激活 "
    "{relu,tanh,gelu})在其上训练产生的「结构优势」签名彼此**可区分**。默认/random 数据"
    "会让签名塌成一团;必须设计好数据让这些杠杆的动力学差异显形。probe 试探、commit 用 "
    "held-out seed 评分:margin 过阈值(可区分)× 效率。"
)


def _fpl_observe_dict(ep) -> dict:
    return {
        "lever_family": ep.config.family,
        "lever_values": lever_values(ep.config.family),
        "budget_steps": ep.config.budget_steps,
        "budget_spent": ep.budget_spent,
        "probe_cost": ep.probe_cost(),
        "correct_margin": ep.config.correct_margin,
        "dsl_schema": DSL_SCHEMA,
        "rules": FPL_RULES,
    }


BBL_RULES = (
    "环境藏了一个真实杠杆的答案(优化器 ∈ 池)。设计探测数据集(bbl-probe),环境返回"
    "池中各答案在你数据上的参考「结构优势」签名 + 黑盒的 mystery 签名;比对 mystery 最"
    "像哪个,用 bbl-guess 猜。默认/random 探测会让签名塌成一团,必须设计好探测让优化器"
    "的动力学差异显形。分数 = 猜对 × 效率。"
)


def _bbl_observe_dict(ep) -> dict:
    return {
        "family": ep.config.family,
        "pool": ep.pool,
        "budget_steps": ep.config.budget_steps,
        "budget_spent": ep.budget_spent,
        "probe_cost": ep.probe_cost(),
        "rules": BBL_RULES,
    }


DX_RULES = (
    "给你一条**病态**训练曲线(sick_curve)—— 但多个病因(lr_too_low / dead_relu / "
    "vanishing_grad)在 loss 曲线上长得几乎一样,光看曲线诊断不出。你要花预算 dx-query "
    "不同的 observable(grad_norm / weight_norm / dead_fraction / per_layer_grad,各不同价),"
    "不同病因在不同 observable 上留签名。查对信号后 dx-answer 定因。分数 = 诊断对 × 效率"
    "(查得越少越高)。地板 = 不查只猜多数病因。"
)


def _dx_observe_dict(ep) -> dict:
    return {
        "sick_curve": ep.sick_curve(),
        "causes": PATHOLOGIES,
        "observables": QUERY_COSTS,
        "budget": ep.config.budget,
        "budget_spent": ep.budget_spent,
        "rules": DX_RULES,
    }


WT_RULES = (
    "N 个候选 config 竞争,你要选出**大尺度**上最优的那个。但你只能花预算在各**尺度**跑"
    "便宜代理实验(wt-run --candidate C --scale s,小尺度便宜、大尺度贵),看该候选在该尺度的值。"
    "关键:差异随尺度**涌现**——小尺度大家挤在一起、纯噪声骗人,大尺度才拉开真差距。信小"
    "尺度会翻车;花预算爬到大尺度验证才可靠。分数 = 大尺度选对 × 效率。"
)


def _wt_observe_dict(wt) -> dict:
    return {
        "candidates": wt.ids,
        "scales": [{"scale": k, "cost": c} for k, c in enumerate(wt.config.scale_costs)],
        "budget": wt.config.budget,
        "budget_spent": wt.budget_spent,
        "rules": WT_RULES,
    }


def _spec(args) -> DatasetSpec:
    return spec_from_args(args.family, args.n_samples, args.modulus, args.n_bits, args.label_noise)


def _add_spec_args(p) -> None:
    p.add_argument("--family", required=True)
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--modulus", type=int, default=7)
    p.add_argument("--n-bits", type=int, default=8)
    p.add_argument("--label-noise", type=float, default=0.0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="architectureiq")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("init", "observe"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", required=True)
    for name in ("probe", "commit"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", required=True)
        _add_spec_args(sp)
    for name in ("tour-init", "tour-observe"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", required=True)
    tri = sub.add_parser("tour-real-init")
    tri.add_argument("--run-dir", required=True)
    tri.add_argument("--task", default="piqa")
    tri.add_argument("--shot", default="zero-shot")
    ta = sub.add_parser("tour-advance")
    ta.add_argument("--run-dir", required=True)
    ta.add_argument("--candidate", required=True)
    ta.add_argument("--steps", type=int, required=True)
    tan = sub.add_parser("tour-answer")
    tan.add_argument("--run-dir", required=True)
    tan.add_argument("--candidate", required=True)
    for name in ("fc-init", "fc-observe"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", required=True)
    fri = sub.add_parser("fc-real-init")
    fri.add_argument("--run-dir", required=True)
    fri.add_argument("--curve", required=True)
    fp = sub.add_parser("fc-predict")
    fp.add_argument("--run-dir", required=True)
    fp.add_argument("--value", type=float, required=True)
    di = sub.add_parser("dr-init")
    di.add_argument("--run-dir", required=True)
    di.add_argument("--pathology", default="too_high")
    do = sub.add_parser("dr-observe")
    do.add_argument("--run-dir", required=True)
    dt = sub.add_parser("dr-treat")
    dt.add_argument("--run-dir", required=True)
    dt.add_argument("--lr", type=float, required=True)
    dc = sub.add_parser("dr-commit")
    dc.add_argument("--run-dir", required=True)
    dc.add_argument("--lr", type=float, required=True)
    dri = sub.add_parser("dr-real-init")
    dri.add_argument("--run-dir", required=True)
    dri.add_argument("--curve", required=True)
    dro = sub.add_parser("dr-real-observe")
    dro.add_argument("--run-dir", required=True)
    drr = sub.add_parser("dr-real-reveal")
    drr.add_argument("--run-dir", required=True)
    drd = sub.add_parser("dr-real-diagnose")
    drd.add_argument("--run-dir", required=True)
    drd.add_argument("--label", required=True)
    bi = sub.add_parser("bb-init")
    bi.add_argument("--run-dir", required=True)
    bi.add_argument("--hidden", default="mlp")
    bo = sub.add_parser("bb-observe")
    bo.add_argument("--run-dir", required=True)
    bp = sub.add_parser("bb-probe")
    bp.add_argument("--run-dir", required=True)
    _add_spec_args(bp)
    bg = sub.add_parser("bb-guess")
    bg.add_argument("--run-dir", required=True)
    bg.add_argument("--arch", required=True)
    fpi = sub.add_parser("fpl-init")
    fpi.add_argument("--run-dir", required=True)
    fpi.add_argument("--lever", default="optimizer")
    fpo = sub.add_parser("fpl-observe")
    fpo.add_argument("--run-dir", required=True)
    for name in ("fpl-probe", "fpl-commit"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", required=True)
        _add_spec_args(sp)
    li = sub.add_parser("bbl-init")
    li.add_argument("--run-dir", required=True)
    li.add_argument("--family", default="optimizer")
    li.add_argument("--hidden", default="sgd")
    lo = sub.add_parser("bbl-observe")
    lo.add_argument("--run-dir", required=True)
    lp = sub.add_parser("bbl-probe")
    lp.add_argument("--run-dir", required=True)
    _add_spec_args(lp)
    lgs = sub.add_parser("bbl-guess")
    lgs.add_argument("--run-dir", required=True)
    lgs.add_argument("--value", required=True)
    wi = sub.add_parser("wt-init")
    wi.add_argument("--run-dir", required=True)
    wi.add_argument("--seed", type=int, default=0)
    wo = sub.add_parser("wt-observe")
    wo.add_argument("--run-dir", required=True)
    wr = sub.add_parser("wt-run")
    wr.add_argument("--run-dir", required=True)
    wr.add_argument("--candidate", required=True)
    wr.add_argument("--scale", type=int, required=True)
    wc = sub.add_parser("wt-commit")
    wc.add_argument("--run-dir", required=True)
    wc.add_argument("--candidate", required=True)
    xi = sub.add_parser("dx-init")
    xi.add_argument("--run-dir", required=True)
    xi.add_argument("--pathology", default="dead_relu")
    xo = sub.add_parser("dx-observe")
    xo.add_argument("--run-dir", required=True)
    xq = sub.add_parser("dx-query")
    xq.add_argument("--run-dir", required=True)
    xq.add_argument("--observable", required=True)
    xa = sub.add_parser("dx-answer")
    xa.add_argument("--run-dir", required=True)
    xa.add_argument("--cause", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "init":
        env = init_state(args.run_dir, default_config())
        print(json.dumps(_observe_dict(env), ensure_ascii=False))
        return 0
    if args.cmd == "observe":
        env = load_state(args.run_dir)
        print(json.dumps(_observe_dict(env), ensure_ascii=False))
        return 0
    if args.cmd == "probe":
        env = load_state(args.run_dir)
        res = env.probe(_spec(args))
        save_state(args.run_dir, env)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "commit":
        env = load_state(args.run_dir)
        res = env.commit(_spec(args))
        save_state(args.run_dir, env)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "tour-init":
        t = init_tournament(args.run_dir, default_tournament_config())
        print(json.dumps(_tour_observe_dict(t), ensure_ascii=False))
        return 0
    if args.cmd == "tour-real-init":
        t = init_real_tournament(args.run_dir, task=args.task, shot=args.shot)
        print(json.dumps(_tour_observe_dict(t), ensure_ascii=False))
        return 0
    if args.cmd == "tour-observe":
        t = load_tournament(args.run_dir)
        print(json.dumps(_tour_observe_dict(t), ensure_ascii=False))
        return 0
    if args.cmd == "tour-advance":
        t = load_tournament(args.run_dir)
        res = t.advance(args.candidate, args.steps)
        save_tournament(args.run_dir, t)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "tour-answer":
        t = load_tournament(args.run_dir)
        res = t.answer(args.candidate)
        save_tournament(args.run_dir, t)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "fc-init":
        ep = init_forecast(args.run_dir, default_forecast_config())
        print(json.dumps(_fc_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "fc-real-init":
        ep = init_real_forecast(args.run_dir, args.curve)
        print(json.dumps(_fc_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "fc-observe":
        ep = load_forecast(args.run_dir)
        print(json.dumps(_fc_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "fc-predict":
        ep = load_forecast(args.run_dir)
        res = ep.predict(args.value)
        save_forecast(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "dr-init":
        ep = init_doctor(args.run_dir, doctor_config(args.pathology))
        print(json.dumps(_dr_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dr-observe":
        ep = load_doctor(args.run_dir)
        print(json.dumps(_dr_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dr-treat":
        ep = load_doctor(args.run_dir)
        res = ep.treat(args.lr)
        save_doctor(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "dr-commit":
        ep = load_doctor(args.run_dir)
        res = ep.commit(args.lr)
        save_doctor(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "dr-real-init":
        ep = init_real_doctor(args.run_dir, args.curve)
        print(json.dumps(_dr_real_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dr-real-observe":
        ep = load_real_doctor(args.run_dir)
        print(json.dumps(_dr_real_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dr-real-reveal":
        ep = load_real_doctor(args.run_dir)
        ep.reveal()
        save_real_doctor(args.run_dir, ep)
        print(json.dumps(_dr_real_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dr-real-diagnose":
        ep = load_real_doctor(args.run_dir)
        res = ep.diagnose(args.label)
        save_real_doctor(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "bb-init":
        ep = init_blackbox(args.run_dir, default_blackbox_config(args.hidden))
        print(json.dumps(_bb_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "bb-observe":
        ep = load_blackbox(args.run_dir)
        print(json.dumps(_bb_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "bb-probe":
        ep = load_blackbox(args.run_dir)
        res = ep.probe(_spec(args))
        save_blackbox(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "bb-guess":
        ep = load_blackbox(args.run_dir)
        res = ep.guess(args.arch)
        save_blackbox(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "fpl-init":
        ep = init_leverfp(args.run_dir, default_leverfp_config(args.lever))
        print(json.dumps(_fpl_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "fpl-observe":
        ep = load_leverfp(args.run_dir)
        print(json.dumps(_fpl_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "fpl-probe":
        ep = load_leverfp(args.run_dir)
        res = ep.probe(_spec(args))
        save_leverfp(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "fpl-commit":
        ep = load_leverfp(args.run_dir)
        res = ep.commit(_spec(args))
        save_leverfp(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "bbl-init":
        ep = init_leverid(args.run_dir, default_leverid_config(args.family, args.hidden))
        print(json.dumps(_bbl_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "bbl-observe":
        ep = load_leverid(args.run_dir)
        print(json.dumps(_bbl_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "bbl-probe":
        ep = load_leverid(args.run_dir)
        res = ep.probe(_spec(args))
        save_leverid(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "bbl-guess":
        ep = load_leverid(args.run_dir)
        res = ep.guess(args.value)
        save_leverid(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "wt-init":
        wt = init_windtunnel(args.run_dir, default_windtunnel_config(args.seed))
        print(json.dumps(_wt_observe_dict(wt), ensure_ascii=False))
        return 0
    if args.cmd == "wt-observe":
        wt = load_windtunnel(args.run_dir)
        print(json.dumps(_wt_observe_dict(wt), ensure_ascii=False))
        return 0
    if args.cmd == "wt-run":
        wt = load_windtunnel(args.run_dir)
        res = wt.run(args.candidate, args.scale)
        save_windtunnel(args.run_dir, wt)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "wt-commit":
        wt = load_windtunnel(args.run_dir)
        res = wt.commit(args.candidate)
        save_windtunnel(args.run_dir, wt)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "dx-init":
        ep = init_diagnostic(args.run_dir, default_diagnostic_config(args.pathology))
        print(json.dumps(_dx_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dx-observe":
        ep = load_diagnostic(args.run_dir)
        print(json.dumps(_dx_observe_dict(ep), ensure_ascii=False))
        return 0
    if args.cmd == "dx-query":
        ep = load_diagnostic(args.run_dir)
        res = ep.query(args.observable)
        save_diagnostic(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    if args.cmd == "dx-answer":
        ep = load_diagnostic(args.run_dir)
        res = ep.answer(args.cause)
        save_diagnostic(args.run_dir, ep)
        print(json.dumps(res.__dict__, ensure_ascii=False))
        return 0
    return 1
