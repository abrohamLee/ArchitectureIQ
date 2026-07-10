# ArchitectureIQ

**An interactive, CPU-only benchmark for testing an LLM agent's *intuition* about neural-network architecture and learning dynamics.**

Inspired by [ARC-AGI-3](https://arcprize.org/) (no answer key, spend a budget, efficiency-scored) and differentiated from [MLS-Bench](https://github.com/Bohan22/MLS-Bench): MLS asks *"can you **implement** a better method?"* — ArchitectureIQ asks *"can you **run the right experiments** to figure something out?"* The skill under test is the **scientific process**, not implementation or recall.

Everything runs on **CPU with tiny models** (the phenomena we test are learning-*dynamics* phenomena that reproduce at tiny scale), so an episode is seconds, deterministic, and cheap.

---

## The litmus: how every task differs from MLS-Bench

> **Can an agent that only *recalls knowledge* — and cannot run experiments — beat the task?**
> If yes → it's an MLS shadow / a knowledge quiz → **cut it**.
> If no (the agent must interactively probe, design a stimulus, or read live dynamics) → **keep it**.

## The universal design contract

Every task must satisfy all five, or it doesn't ship:

1. **Actions mirror a real researcher's workflow** — diagnosis = a triage ladder (cheap reads → hypothesis → one expensive confirming experiment); forecasting = the early-stopping decision; selection = a scaling ladder. Not Q&A — observe → hypothesize → pay to experiment → conclude.
2. **Three-layer grounding** — *real replay* (frozen Pythia/Marin curves & observables — un-memorizable, but passive), *precomputed replay* (offline-generated realistic-scale runs), and *tiny live-sim* (the environment owns a fully-instrumented training loop the agent can actually experiment on). The investigable layer is the sim; real data validates that the sim is faithful.
3. **Efficiency is intrinsic to the score** — `score = correctness × min(1, expert_budget / agent_budget)²`. Getting the right answer by brute-force (query everything, retrain many times) is squared-penalized. This is the main discriminator.
4. **Three anti-memorization gates** — an instance is only valid if (a) a *closed-book, config-only* agent **fails** it (else it's recall → cut), (b) matched / boundary instances turn a memorized answer into a *confidently-wrong* trap, and (c) an agent that genuinely understands the mechanism **succeeds** (else it's luck / chaos).
5. **MLS differentiation** — you never edit training code to *fix* anything; you observe, intervene, or predict with a limited, priced instrument set.

## Scoring, in plain terms

- **RHAE-ML** (most tasks): `1[correct] × min(1, human/agent)²`, range **0–1**. Wrong = 0; right-and-efficient = 1.
- **Skill** (forecast): `1 − agent_error / best-naive-baseline_error`, range **(−∞, 1]**. 1 = perfect, 0 = tied the naive baseline, **negative = worse than a straight line**. These two scales don't mix — read the leaderboard per-task.

---

## The tasks

Four "scientist" tasks, each grounded in a real research workflow. (See `docs/` locally for the full design spec.)

| Task | What it tests | Real-researcher analogy | Status |
|---|---|---|---|
| **Diagnostic** | query the right observables to find an architectural fault | debugging a sick training run (grad-norm/activation triage → intervene) | designed + toy engine built |
| **Forecast** | predict a training curve's far future from its early part | the early-stopping / run-continuation decision | **buy-prefix engine built, real-data eval run** |
| **Wind Tunnel** | pick the best config to scale up from cheap proxy experiments | μP / scaling-ladder hyperparameter selection | designed + toy engine built |
| **Jenga** *(pending)* | find the minimal architecture by ablating components | ablation studies — what is each component *for*? | design pending (A/B) |

The pathology pool for **Diagnostic** is grounded in *real documented incidents* — Marin's 8B `lm_head`-norm explosion (→ z-loss), the 32B `update-norm → grad-norm → loss-spike` ordering (→ clip), Muon divergence, Pythia-1b's fp16 spike — reproduced in a controlled tiny sim where we own every observable and can verify a proposed fix by actually retraining.

Earlier iterations also built five "v0" tasks (fingerprint, doctor, tournament, blackbox + real-lever tiers) and a Harbor packaging; the first real measurement showed them near-saturated, which drove the researcher-grounded redesign above. That history is in the git log and `docs/`.

---

## Quickstart

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync                       # install (Python 3.11+, torch CPU, numpy)
uv run pytest -q              # run the test suite

# Play the forecast task on a real Pythia emergence curve:
uv run python -m architectureiq bpfc-init    --run-dir /tmp/g --index 0
uv run python -m architectureiq bpfc-reveal  --run-dir /tmp/g --until 20000
uv run python -m architectureiq bpfc-predict --run-dir /tmp/g --value 0.62
```

Every task is a set of typed CLI subcommands (`<task>-init / -observe / -probe|reveal|query / -commit|predict|answer`). The environment owns the training loop, so the agent can only interact through the priced action interface — no reward hacking.

---

## Repo structure

```
src/architectureiq/   task engines + environment (models, trainer, telemetry, scoring, CLI)
tests/                per-task TDD tests
tasks/                agent-facing task manuals
data/real_curves/     harvested real training curves (Pythia eval-acc, Marin loss)
scripts/              data harvesting (fetch_*) + agent eval harnesses (eval_*) + plots
harbor/               Harbor dataset packaging (CPU-only)
docs/                 design specs (git-ignored; kept local)
```

## Results so far (honest)

Two real measurements against frontier agents (claude-code = Opus/Sonnet, codex = GPT-5.5 at low/med/high effort):

- **First run (v0 real tiers, correctness-only):** ~4/5 configs scored 100% — **saturated**. The cause was *our measurement*: we had dropped the efficiency term, picked easy fixed instances, and excluded the hardest task. This triggered the redesign.
- **Forecast on real Pythia emergence curves:** all configs beat the naive baseline **~75–80%** with positive skill (+0.4 to +0.58) — so *emergence* prediction is largely within frontier intuition. The discriminating instances are the **anti-intuition** ones (plateaus that don't break, non-monotonic drops, matched pairs) where the "it keeps emerging" prior is wrong. *(Note: the eval was not sandboxed from the data files; a few high-effort codex runs looked up answers — those exact-match rows are filtered out. A container-sandboxed re-run is the fix.)*

The recurring lesson: **restore efficiency scoring + use anti-intuition instances + let the config-only gate cut anything recall solves.**

## Data sources

- **Pythia** — cross-scale eval-accuracy across 10 model sizes (emergence / scaling).
- **Marin** — real training-loss sweeps (optimizer / batch / LR) with genuine crossovers, plus flagship 8B/32B runs logging ~3500 per-layer per-component grad/param norms (the diagnostic observable schema).

## Status

Research-in-progress. The design has converged on the four researcher-grounded tasks above; the build is mid-flight (forecast furthest along). Not a finished, calibrated benchmark yet.
