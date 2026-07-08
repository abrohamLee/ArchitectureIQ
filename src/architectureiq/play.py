import argparse
import json

from architectureiq.datasets import DatasetSpec
from architectureiq.episode import default_config
from architectureiq.runstate import init_state, load_state, save_state
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
    return 1
