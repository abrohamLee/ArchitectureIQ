#!/usr/bin/env python3
"""实测 harness:让 claude-code / codex 无头玩真实 tier,独立评分 + 记成本。

- agent 在仓库外的干净 workdir 用 venv CLI 玩(禁读源码/state),把答案写进 submission.json。
- verifier(architectureiq.harbor.score_submission)拿隐藏答案独立重算 0/1。
- 成本:claude -p --output-format json 直接给 total_cost_usd;两者都抓 token,用参考单价
  统一折算 $(便于 apples-to-apples,和 ARC-AGI-2 一样按 API 成本画),单价见 PRICES(估值,可改)。

用法: uv run python scripts/eval_agents.py            # 跑全部
       uv run python scripts/eval_agents.py --smoke   # 只跑 opus×blackbox 验证管线
"""
import argparse
import json
import os
import re
import subprocess
import time

from architectureiq.harbor import score_submission

VENV_PY = "/tmp/aiq_eval_venv/bin/python"
EVAL_ROOT = "/tmp/aiq_eval"
RESULTS = os.path.join(EVAL_ROOT, "results.jsonl")

# 参考 API 单价 ($/百万 token) — 估值,画图用统一口径;可按真实定价调整。
PRICES = {
    "opus":    {"in": 5.0,  "out": 25.0},
    "sonnet":  {"in": 3.0,  "out": 15.0},
    "gpt-5.5": {"in": 1.25, "out": 10.0},
}

CONFIGS = [
    {"name": "Opus",          "harness": "claude", "model": "opus",    "price": "opus"},
    {"name": "Sonnet",        "harness": "claude", "model": "sonnet",  "price": "sonnet"},
    {"name": "GPT-5.5 (low)", "harness": "codex",  "effort": "low",    "price": "gpt-5.5"},
    {"name": "GPT-5.5 (med)", "harness": "codex",  "effort": "medium", "price": "gpt-5.5"},
    {"name": "GPT-5.5 (high)","harness": "codex",  "effort": "high",   "price": "gpt-5.5"},
]

TASKS = {
    "fingerprint_lever": {
        "setup": ["fpl-init", "--lever", "optimizer"],
        "hidden": {"task_id": "fingerprint_lever", "lever_family": "optimizer",
                   "correct_margin": 0.22, "ref_seeds": [10, 11, 12],
                   "query_seeds": [13, 14, 15], "probe_steps": 80},
        "goal": "设计一个数据集,让优化器 {adam,sgd,rmsprop} 的「结构优势」签名彼此可区分(margin≥0.22);random 数据会塌陷。",
        "cmds": ("fpl-observe --run-dir game\n"
                 "fpl-probe --run-dir game --family <modular_addition|parity|random> [--n-samples N --modulus M]"),
        "submission": {"family": "modular_addition", "n_samples": 300, "modulus": 7},
    },
    "blackbox_lever": {
        "setup": ["bbl-init", "--family", "optimizer", "--hidden", "sgd"],
        "hidden": {"task_id": "blackbox_lever", "lever_family": "optimizer", "hidden_value": "sgd"},
        "goal": "环境藏了一个优化器(∈{adam,sgd,rmsprop})。设计探测数据集让它显形,认出它。",
        "cmds": ("bbl-observe --run-dir game\n"
                 "bbl-probe --run-dir game --family <modular_addition|parity|random> [--n-samples N --modulus M]"),
        "submission": {"guess": "adam"},
    },
    "diagnostic": {
        "setup": ["dx-init", "--pathology", "vanishing_grad"],
        "hidden": {"task_id": "diagnostic", "pathology": "vanishing_grad"},
        "goal": ("给你一条病态训练曲线,病因∈{lr_too_low,dead_relu,vanishing_grad},loss 曲线看不出。"
                 "查 observable(不同病因在不同 observable 上留签名)定因。"),
        "cmds": ("dx-observe --run-dir game\n"
                 "dx-query --run-dir game --observable <grad_norm|weight_norm|dead_fraction|per_layer_grad>"),
        "submission": {"cause": "lr_too_low"},
    },
    "windtunnel": {
        "setup": ["wt-init", "--seed", "3"],
        "hidden": {"task_id": "windtunnel", "seed": 3, "regret_threshold": 0.05},
        "goal": ("N 个候选竞争,选出大尺度上最优的。差异随尺度涌现——小尺度纯噪声骗人。"
                 "花预算在各尺度跑代理实验,别信小尺度。"),
        "cmds": ("wt-observe --run-dir game\n"
                 "wt-run --run-dir game --candidate <C> --scale <k>"),
        "submission": {"candidate": "A"},
    },
}

PROMPT = """你在玩一个交互式实验设计游戏。规则:只能用下面列出的 CLI 子命令收集信息;\
**禁止**读取任何 *_state.json 文件或 architectureiq 的 python 源码(那是作弊)。

运行 CLI 的方式(game 目录已开局,不要再 init):
  {py} -m architectureiq <上面的子命令>

任务目标:{goal}

可用命令:
  {py} -m architectureiq {cmds}

玩法:先 observe 看规则和预算,再用 probe/query/run 花预算做实验收集证据,推理后给出答案。

完成后**必须**把最终答案写进当前目录的 submission.json 文件,格式严格如下(键名照抄,值换成你的答案):
  {submission}

现在开始。最后确保 submission.json 已写好。"""


def _tok_scan(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = k.lower()
            if "input_tokens" in lk and isinstance(v, (int, float)):
                acc["in"] = max(acc["in"], int(v))
            elif "output_tokens" in lk and isinstance(v, (int, float)):
                acc["out"] = max(acc["out"], int(v))
            else:
                _tok_scan(v, acc)
    elif isinstance(obj, list):
        for x in obj:
            _tok_scan(x, acc)


def run_claude(prompt, model, workdir, env, timeout):
    cmd = ["claude", "-p", prompt, "--model", model,
           "--output-format", "json", "--dangerously-skip-permissions"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                       text=True, timeout=timeout)
    wall = time.time() - t0
    tin = tout = 0
    real_cost = None
    try:
        obj = json.loads(p.stdout)
        real_cost = obj.get("total_cost_usd")
        u = obj.get("usage", {})
        tin = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        tout = u.get("output_tokens", 0)
    except Exception:
        pass
    return tin, tout, real_cost, wall, p.stdout[-500:] + p.stderr[-500:]


def run_codex(prompt, effort, workdir, env, timeout):
    cmd = ["codex", "exec", "-c", f"model_reasoning_effort={effort}",
           "--json", "--sandbox", "workspace-write", "--skip-git-repo-check", prompt]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True,
                       text=True, timeout=timeout)
    wall = time.time() - t0
    acc = {"in": 0, "out": 0}
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                _tok_scan(json.loads(line), acc)
            except Exception:
                pass
    return acc["in"], acc["out"], None, wall, p.stdout[-500:] + p.stderr[-500:]


def one_run(cfg, task_id, timeout):
    task = TASKS[task_id]
    workdir = os.path.join(EVAL_ROOT, cfg["name"].replace(" ", "_").replace("(", "").replace(")", ""), task_id)
    os.makedirs(workdir, exist_ok=True)
    game = os.path.join(workdir, "game")
    # 干净开局
    subprocess.run([VENV_PY, "-m", "architectureiq"] + task["setup"] + ["--run-dir", game],
                   capture_output=True, text=True)
    sub_path = os.path.join(workdir, "submission.json")
    if os.path.exists(sub_path):
        os.remove(sub_path)

    prompt = PROMPT.format(py=VENV_PY, goal=task["goal"], cmds=task["cmds"],
                           submission=json.dumps(task["submission"], ensure_ascii=False))
    env = dict(os.environ)
    env["PATH"] = "/tmp/aiq_eval_venv/bin:" + env.get("PATH", "")

    err = None
    try:
        if cfg["harness"] == "claude":
            tin, tout, real_cost, wall, tail = run_claude(prompt, cfg["model"], workdir, env, timeout)
        else:
            tin, tout, real_cost, wall, tail = run_codex(prompt, cfg["effort"], workdir, env, timeout)
    except subprocess.TimeoutExpired:
        tin = tout = 0; real_cost = None; wall = timeout; tail = "TIMEOUT"

    # 读 submission + 独立评分
    submission = {}
    if os.path.exists(sub_path):
        try:
            with open(sub_path) as f:
                submission = json.load(f)
        except Exception as e:
            err = f"bad submission.json: {e}"
    else:
        err = "no submission.json"
    try:
        reward = float(score_submission(task_id, task["hidden"], submission)) if submission else 0.0
    except Exception as e:
        reward = 0.0; err = f"score error: {e}"

    price = PRICES[cfg["price"]]
    est_cost = (tin / 1e6) * price["in"] + (tout / 1e6) * price["out"]
    rec = {"config": cfg["name"], "harness": cfg["harness"], "task": task_id,
           "reward": reward, "tokens_in": tin, "tokens_out": tout,
           "est_cost_usd": round(est_cost, 5), "real_cost_usd": real_cost,
           "wall_s": round(wall, 1), "submission": submission, "error": err,
           "tail": tail[-300:]}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{cfg['name']:14s} · {task_id:18s}] reward={reward} "
          f"tok={tin}/{tout} ${est_cost:.4f} {wall:.0f}s {err or ''}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timeout", type=int, default=420)
    args = ap.parse_args()
    os.makedirs(EVAL_ROOT, exist_ok=True)

    if args.smoke:
        one_run(CONFIGS[0], "blackbox_lever", args.timeout)
        return
    for cfg in CONFIGS:
        for task_id in TASKS:
            one_run(cfg, task_id, args.timeout)
    print("== done ==")


if __name__ == "__main__":
    main()
