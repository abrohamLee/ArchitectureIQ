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

    pt = {}  # cfg -> (cost, score)
    for cfg, a in agg.items():
        n = a["n"] or 1
        pt[cfg] = (sum(a["cost"]) / n, 100.0 * sum(a["reward"]) / n)

    # GPT-5.5 的 effort 阶梯用虚线连起来(同模型、成本递增),像 ARC 榜
    ladder = [c for c in ["GPT-5.5 (low)", "GPT-5.5 (med)", "GPT-5.5 (high)"] if c in pt]
    if len(ladder) > 1:
        xs = [pt[c][0] for c in ladder]; ys = [pt[c][1] for c in ladder]
        ax.plot(xs, ys, "--", color="#3B78C4", alpha=0.5, zorder=2, linewidth=1.2)

    # 标签偏移(避免 100% 处堆叠):(dx_pts, dy_pts, ha, va)
    LABEL = {
        "GPT-5.5 (low)":  (0, 12, "center", "bottom"),
        "GPT-5.5 (med)":  (0, -16, "center", "top"),
        "GPT-5.5 (high)": (8, 10, "left", "bottom"),
        "Sonnet":         (10, 6, "left", "bottom"),
        "Opus":           (10, 0, "left", "center"),
    }
    for cfg, (cost, score) in pt.items():
        color = COLOR.get(cfg, "#888")
        ax.scatter([cost], [score], s=190, color=color, zorder=4,
                   edgecolors="white", linewidths=0.7)
        dx, dy, ha, va = LABEL.get(cfg, (8, 0, "left", "center"))
        ax.annotate(cfg, (cost, score), color=color, fontsize=11.5, fontweight="bold",
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va)

    ax.set_xscale("log")
    ax.set_xlabel("COST PER TASK ($, est. API price)", fontsize=13, labelpad=10)
    ax.set_ylabel("SCORE (%)", fontsize=13, labelpad=10)
    ax.set_ylim(-5, 105)
    ax.set_title("ArchitectureIQ Leaderboard — real tiers (4 tasks x 1 episode)",
                 fontsize=17, pad=18, color="#f2f2f2")
    ax.grid(True, which="both", alpha=0.12, color="#666")
    ax.set_axisbelow(True)
    fig.text(0.5, 0.02,
             "codex = GPT-5.5 (low/med/high)  ·  claude-code = Opus / Sonnet  ·  "
             "cost estimated from tokens x reference price  ·  score = mean correctness over 4 tasks",
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
