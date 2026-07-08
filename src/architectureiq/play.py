import argparse
import json

from architectureiq.datasets import DatasetSpec
from architectureiq.episode import default_config
from architectureiq.runstate import init_state, load_state, save_state

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
    return 1
