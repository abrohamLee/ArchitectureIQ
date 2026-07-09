#!/usr/bin/env python3
"""预测任务实测 → ARC-AGI-2 风格散点。y=击败基线的实例占比(%),x=每实例成本。"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/tmp/aiq_fc_eval/results.jsonl"
OUT = "/Users/abroham/Documents/ArchitectureIQ/forecast_leaderboard.png"
COLOR = {"Opus": "#E8663D", "Sonnet": "#F0A05A",
         "GPT-5.5 (low)": "#4A90E2", "GPT-5.5 (med)": "#3B78C4", "GPT-5.5 (high)": "#2C5AA6"}


def main():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    agg = defaultdict(lambda: {"beat": [], "cost": [], "skill": []})
    for r in rows:
        a = agg[r["config"]]
        a["beat"].append(r["beat"]); a["cost"].append(r["est_cost_usd"]); a["skill"].append(r["skill_clip"])

    plt.rcParams.update({"figure.facecolor": "#0d0d0f", "axes.facecolor": "#0d0d0f",
        "text.color": "#e8e8e8", "axes.labelcolor": "#e8e8e8", "xtick.color": "#b0b0b0",
        "ytick.color": "#b0b0b0", "axes.edgecolor": "#333", "font.size": 12})
    fig, ax = plt.subplots(figsize=(11, 7))

    pt = {c: (sum(a["cost"])/len(a["cost"]), 100*sum(a["beat"])/len(a["beat"])) for c, a in agg.items()}
    ladder = [c for c in ["GPT-5.5 (low)", "GPT-5.5 (med)", "GPT-5.5 (high)"] if c in pt]
    if len(ladder) > 1:
        ax.plot([pt[c][0] for c in ladder], [pt[c][1] for c in ladder], "--", color="#3B78C4", alpha=0.5, zorder=2)
    LAB = {"GPT-5.5 (low)": (0,12,"center","bottom"), "GPT-5.5 (med)": (0,-16,"center","top"),
           "GPT-5.5 (high)": (8,10,"left","bottom"), "Sonnet": (10,6,"left","bottom"), "Opus": (10,0,"left","center")}
    for c, (cost, beat) in pt.items():
        col = COLOR.get(c, "#888")
        ax.scatter([cost], [beat], s=190, color=col, zorder=4, edgecolors="white", linewidths=0.7)
        dx, dy, ha, va = LAB.get(c, (8,0,"left","center"))
        ms = sum(agg[c]["skill"])/len(agg[c]["skill"])
        ax.annotate(f"{c}  (skill {ms:+.2f})", (cost, beat), color=col, fontsize=11, fontweight="bold",
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va)

    ax.axhline(0, color="#444", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("COST PER INSTANCE ($, est. API price)", fontsize=13, labelpad=10)
    ax.set_ylabel("% INSTANCES BEATING NAIVE BASELINE", fontsize=13, labelpad=10)
    ax.set_ylim(-5, 105)
    ax.set_title("ArchitectureIQ · Forecast — real emergence curves (20 instances)", fontsize=16, pad=18, color="#f2f2f2")
    ax.grid(True, which="both", alpha=0.12, color="#666"); ax.set_axisbelow(True)
    fig.text(0.5, 0.02, "predict real Pythia eval-acc endpoint from early curve  ·  "
             "baseline = best of {linear, power-law, persistence}  ·  skill = mean over instances (clipped)",
             ha="center", color="#888", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT, dpi=150, facecolor=fig.get_facecolor())
    print("saved", OUT)
    print(f"\n{'配置':16s} {'击败基线%':>9s} {'平均skill':>9s} {'$/实例':>9s}")
    for c, a in sorted(agg.items(), key=lambda kv: -sum(kv[1]['beat'])/len(kv[1]['beat'])):
        n = len(a["beat"])
        print(f"{c:16s} {100*sum(a['beat'])/n:>8.0f} {sum(a['skill'])/n:>+9.2f} {sum(a['cost'])/n:>9.4f}")


if __name__ == "__main__":
    main()
