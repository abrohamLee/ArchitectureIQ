import argparse
import json

from architectureiq.datasets import DatasetSpec
from architectureiq.episode import default_config
from architectureiq.runstate import init_state, load_state, save_state
from architectureiq.blackbox import default_blackbox_config
from architectureiq.blackboxstate import init_blackbox, load_blackbox, save_blackbox
from architectureiq.doctor import doctor_config
from architectureiq.doctorstate import init_doctor, load_doctor, save_doctor
from architectureiq.forecast import default_forecast_config
from architectureiq.forecaststate import (
    init_forecast,
    load_forecast,
    save_forecast,
)
from architectureiq.tournament import default_tournament_config
from architectureiq.tourstate import (
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
    return 1
