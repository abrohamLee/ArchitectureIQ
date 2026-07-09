#!/usr/bin/env python3
"""实测 harness · 预测任务:agent 看配置+前缀,预测真实曲线远期值,skill 评分。

固定揭示到 reveal 封顶(去掉 reveal 决策,纯"从早期预测远期");skill vs 最优朴素基线。
y 指标 = 击败基线的实例占比(skill>0)+ 平均 clipped skill。
用法: uv run python scripts/eval_forecast.py [--n 20]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根

from architectureiq.bpforecast import (
    BPForecastEpisode,
    load_all_real,
    strongest_baseline,
)
from scripts.eval_agents import PRICES, run_claude, run_codex

EVAL_ROOT = "/tmp/aiq_fc_eval"
RESULTS = os.path.join(EVAL_ROOT, "results.jsonl")

CONFIGS = [
    {"name": "Opus",           "harness": "claude", "model": "opus",   "price": "opus"},
    {"name": "Sonnet",         "harness": "claude", "model": "sonnet", "price": "sonnet"},
    {"name": "GPT-5.5 (low)",  "harness": "codex",  "effort": "low",    "price": "gpt-5.5"},
    {"name": "GPT-5.5 (med)",  "harness": "codex",  "effort": "medium", "price": "gpt-5.5"},
    {"name": "GPT-5.5 (high)", "harness": "codex",  "effort": "high",   "price": "gpt-5.5"},
]

PROMPT = """你是训练动力学专家。下面是一条**真实**训练曲线的**早期段**,预测它在远期某步的指标值。

配置(架构/数据/指标):
{config}
指标: {metric}   要预测的步: {horizon}

早期曲线(step -> value),这是你能看到的全部:
{prefix}

任务:预测第 {horizon} 步的 {metric} 值。关键——别只顺着曲线线性外推;要判断**这种架构/训练接下来会怎样**:
继续改善?平台?突然涌现/跃升?发散?据此给出你的预测。

把预测写进当前目录的 submission.json,格式严格:
  {{"value": <你的预测数值>}}
只需一个数。现在推理并写文件。"""


def one_run(cfg, inst, idx, timeout):
    ep = BPForecastEpisode(inst)
    ep.reveal(inst.steps[-1])  # 揭示到封顶
    idxc = ep._idx_of(ep.revealed_until_step)
    prefix = "\n".join(f"  {s} -> {round(v,4)}" for s, v in zip(inst.steps[:idxc+1], inst.values[:idxc+1]))
    prompt = PROMPT.format(config=json.dumps(inst.config, ensure_ascii=False),
                           metric=inst.metric, horizon=ep.horizon_step(), prefix=prefix)
    workdir = os.path.join(EVAL_ROOT, cfg["name"].replace(" ", "_").replace("(", "").replace(")", ""), f"inst{idx}")
    os.makedirs(workdir, exist_ok=True)
    sub = os.path.join(workdir, "submission.json")
    if os.path.exists(sub):
        os.remove(sub)
    env = dict(os.environ); env["PATH"] = "/tmp/aiq_eval_venv/bin:" + env.get("PATH", "")
    try:
        if cfg["harness"] == "claude":
            tin, tout, _, wall, tail = run_claude(prompt, cfg["model"], workdir, env, timeout)
        else:
            tin, tout, _, wall, tail = run_codex(prompt, cfg["effort"], workdir, env, timeout)
    except Exception as e:
        tin = tout = 0; wall = timeout; tail = f"ERR {e}"

    value = None; err = None
    if os.path.exists(sub):
        try:
            value = float(json.load(open(sub))["value"])
        except Exception as e:
            err = f"bad submission: {e}"
    else:
        err = "no submission"
    if value is None:
        skill = -1.0; res = None
    else:
        res = ep.predict(value)
        skill = res.skill
    price = PRICES[cfg["price"]]
    cost = (tin/1e6)*price["in"] + (tout/1e6)*price["out"]
    rec = {"config": cfg["name"], "task_idx": idx, "inst": inst.id, "metric": inst.metric,
           "skill": skill, "skill_clip": max(-1.0, min(1.0, skill)), "beat": 1 if skill > 0 else 0,
           "value": value, "truth": res.truth if res else None, "baseline": res.baseline_pred if res else None,
           "tokens_in": tin, "tokens_out": tout, "est_cost_usd": round(cost,5), "wall_s": round(wall,1), "error": err}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{cfg['name']:14s} · inst{idx:02d}] skill={skill:+.2f} beat={rec['beat']} "
          f"pred={value} truth={rec['truth']} base={rec['baseline']} ${cost:.3f} {wall:.0f}s {err or ''}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(EVAL_ROOT, exist_ok=True)

    def abs_base_err(inst):
        ep = BPForecastEpisode(inst); ep.reveal(inst.steps[-1]); k = ep._idx_of(ep.revealed_until_step)
        _, base = strongest_baseline(inst.steps[:k+1], inst.values[:k+1], inst.steps[-1], inst.values[-1])
        return abs(base - inst.values[-1])

    cands = [i for i in load_all_real("data/real_curves")
             if i.metric in ("acc", "acc_norm") and (max(i.values) - min(i.values)) >= 0.12]
    bank = sorted(cands, key=lambda i: -abs_base_err(i))[:args.n]  # 真涌现 + 基线挂最狠
    print(f"评测 {len(bank)} 个真涌现实例 × {len(CONFIGS)} 配置", flush=True)
    for cfg in CONFIGS:
        for idx, inst in enumerate(bank):
            one_run(cfg, inst, idx, args.timeout)
    print("== done ==")


if __name__ == "__main__":
    main()
