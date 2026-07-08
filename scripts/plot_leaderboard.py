#!/usr/bin/env python3
"""把 eval 结果画成 ARC-AGI-2 风格的 cost-vs-score 散点。

用法: uv run python scripts/plot_leaderboard.py
读 /tmp/aiq_eval/results.jsonl → 每个配置聚合(mean reward, mean $/task)→ 输出 leaderboard.png
"""
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/tmp/aiq_eval/results.jsonl"
OUT = "/Users/abroham/Documents/ArchitectureIQ/leaderboard.png"

# 配置 → 颜色(claude=橙红,GPT=蓝,像 ARC 榜按 provider 上色)
COLOR = {
    "Opus": "#E8663D", "Sonnet": "#F0A05A",
    "GPT-5.5 (low)": "#4A90E2", "GPT-5.5 (med)": "#3B78C4", "GPT-5.5 (high)": "#2C5AA6",
}


def main():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    agg = defaultdict(lambda: {"reward": [], "cost": [], "n": 0})
    for r in rows:
        a = agg[r["config"]]
        a["reward"].append(r["reward"])
        a["cost"].append(r["est_cost_usd"])
        a["n"] += 1

    plt.rcParams.update({
        "figure.facecolor": "#0d0d0f", "axes.facecolor": "#0d0d0f",
        "text.color": "#e8e8e8", "axes.labelcolor": "#e8e8e8",
        "xtick.color": "#b0b0b0", "ytick.color": "#b0b0b0",
        "axes.edgecolor": "#333", "font.size": 12,
    })
    fig, ax = plt.subplots(figsize=(11, 7))

    for cfg, a in agg.items():
        n = a["n"] or 1
        score = 100.0 * sum(a["reward"]) / n
        cost = sum(a["cost"]) / n  # 每任务平均 $
        color = COLOR.get(cfg, "#888")
        ax.scatter([cost], [score], s=180, color=color, zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(f"  {cfg}", (cost, score), color=color, fontsize=11,
                    va="center", ha="left", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("每任务成本  COST PER TASK ($, 参考 API 单价)", fontsize=13, labelpad=10)
    ax.set_ylabel("得分  SCORE (%)", fontsize=13, labelpad=10)
    ax.set_ylim(-5, 105)
    ax.set_title("ArchitectureIQ Leaderboard —— 真实 tier(4 任务 × 1 局)",
                 fontsize=17, pad=18, color="#f2f2f2")
    ax.grid(True, which="both", alpha=0.12, color="#666")
    ax.set_axisbelow(True)
    fig.text(0.5, 0.02,
             "codex=GPT-5.5(low/med/high) · claude-code=Opus/Sonnet · 成本按参考单价估算 · 得分=4 任务正确率均值",
             ha="center", color="#888", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
    print("saved", OUT)
    # 附:文字榜
    print(f"\n{'配置':16s} {'得分%':>7s} {'$/任务':>9s} {'局数':>5s}")
    for cfg, a in sorted(agg.items(), key=lambda kv: -sum(kv[1]['reward'])/(kv[1]['n'] or 1)):
        n = a["n"] or 1
        print(f"{cfg:16s} {100*sum(a['reward'])/n:>6.1f} {sum(a['cost'])/n:>9.4f} {n:>5d}")


if __name__ == "__main__":
    main()
