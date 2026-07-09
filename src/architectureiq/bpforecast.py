"""② 预测 · 买前缀版(buy-prefix forecast)—— 见 spec 2026-07-09 §3。

agent 看架构+配置+一小段免费前缀,可花 reveal 揭示更多曲线,押**远期(F 步)**的值。
评分 = skill vs "同一段前缀上最优朴素基线"(线性/幂律/persistence):
  skill = 1 − |V_agent − 真值_F| / |V_baseline − 真值_F|
自调节:揭示过了拐点 → 基线也会预测 → skill→0(看过头自罚)。skill 可为负。
"""
import json
import math
import os
from dataclasses import dataclass


# ---------- 朴素基线(在揭示的前缀上拟合,外推到 F)----------

def _persistence(steps, values, target_step):
    return values[-1]


def _linear(steps, values, target_step):
    n = len(steps)
    if n < 2:
        return values[-1]
    mx = sum(steps) / n
    my = sum(values) / n
    denom = sum((s - mx) ** 2 for s in steps)
    if denom == 0:
        return values[-1]
    slope = sum((s - mx) * (v - my) for s, v in zip(steps, values)) / denom
    return my + slope * (target_step - mx)


def _power_law(steps, values, target_step):
    # 对 (log step, log|value|) 做线性拟合 → v ≈ sign * exp(b) * step^a;非正/退化则回退线性
    pts = [(s, v) for s, v in zip(steps, values) if s > 0 and v != 0]
    if len(pts) < 2:
        return _linear(steps, values, target_step)
    sign = 1.0 if pts[-1][1] > 0 else -1.0
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(abs(v)) for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0 or target_step <= 0:
        return _linear(steps, values, target_step)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    b = my - a * mx
    return sign * math.exp(b + a * math.log(target_step))


_BASELINES = {"persistence": _persistence, "linear": _linear, "power_law": _power_law}


def _rmse_on_prefix(fn, steps, values):
    # 用留一式的粗略残差:拟合全前缀,量在前缀点上的拟合误差(选前缀内拟合最好的方法,不偷看真值)
    err = 0.0
    for i in range(len(steps)):
        pred = fn(steps, values, steps[i])
        err += (pred - values[i]) ** 2
    return math.sqrt(err / len(steps))


def baseline_predictions(steps, values, target_step):
    """三个朴素方法各自外推到 target_step,返回 {方法名: 预测值}。"""
    return {name: fn(steps, values, target_step) for name, fn in _BASELINES.items()}


def best_baseline(steps, values, target_step):
    """选在揭示前缀上拟合最好的朴素方法(不偷看真值),外推。返回 (方法名, 预测值)。"""
    best_name, best_res = None, float("inf")
    for name, fn in _BASELINES.items():
        res = _rmse_on_prefix(fn, steps, values)
        if res < best_res:
            best_name, best_res = name, res
    return best_name, _BASELINES[best_name](steps, values, target_step)


def strongest_baseline(steps, values, target_step, truth):
    """'最优朴素基线' = 三个方法里对真值误差最小的那个(agent 要打败它)。"""
    preds = baseline_predictions(steps, values, target_step)
    name = min(preds, key=lambda m: abs(preds[m] - truth))
    return name, preds[name]


def skill(agent_pred, baseline_pred, truth):
    base_err = abs(baseline_pred - truth)
    if base_err < 1e-9:
        return 0.0 if abs(agent_pred - truth) > 1e-9 else 1.0
    return 1.0 - abs(agent_pred - truth) / base_err


# ---------- 实例 + 引擎 ----------

@dataclass
class FCInstance:
    id: str
    config: dict            # 架构/配置元数据(observe 时公开)
    metric: str
    steps: list
    values: list
    free_frac: float = 0.10   # 免费前缀比例
    reveal_cap_frac: float = 0.60  # 最多能揭示到的比例(远期在其外)


@dataclass
class RevealResult:
    steps: list
    values: list
    revealed_until: int
    at_cap: bool


@dataclass
class PredictResult:
    skill: float
    agent_pred: float
    baseline_pred: float
    baseline_method: str
    truth: float
    target_step: int
    revealed_until_step: int


class BPForecastEpisode:
    def __init__(self, instance: FCInstance, revealed_until_step: int | None = None):
        self.inst = instance
        n = len(instance.steps)
        self._free_idx = max(1, int(n * instance.free_frac))
        self._cap_idx = max(self._free_idx, int(n * instance.reveal_cap_frac))
        self._horizon_idx = n - 1
        # 已揭示到的 step(默认=免费前缀末)
        if revealed_until_step is None:
            self.revealed_until_step = instance.steps[self._free_idx - 1]
        else:
            self.revealed_until_step = revealed_until_step

    def _idx_of(self, step):
        # 揭示到 <= step 的最后一个索引
        idx = 0
        for i, s in enumerate(self.inst.steps):
            if s <= step:
                idx = i
        return idx

    def horizon_step(self):
        return self.inst.steps[self._horizon_idx]

    def observe(self) -> dict:
        idx = self._idx_of(self.revealed_until_step)
        return {
            "config": self.inst.config,
            "metric": self.inst.metric,
            "horizon_step": self.horizon_step(),
            "reveal_cap_step": self.inst.steps[self._cap_idx],
            "prefix_steps": self.inst.steps[: idx + 1],
            "prefix_values": self.inst.values[: idx + 1],
        }

    def reveal(self, until_step: int) -> RevealResult:
        cap_step = self.inst.steps[self._cap_idx]
        at_cap = until_step >= cap_step
        target = min(until_step, cap_step)
        self.revealed_until_step = max(self.revealed_until_step, target)
        idx = self._idx_of(self.revealed_until_step)
        return RevealResult(self.inst.steps[: idx + 1], self.inst.values[: idx + 1],
                            self.revealed_until_step, at_cap)

    def predict(self, value: float, target_step: int | None = None) -> PredictResult:
        F = self.horizon_step() if target_step is None else target_step
        truth = self.inst.values[self._horizon_idx]
        idx = self._idx_of(self.revealed_until_step)
        ps, pv = self.inst.steps[: idx + 1], self.inst.values[: idx + 1]
        method, base_pred = strongest_baseline(ps, pv, F, truth)
        return PredictResult(skill(value, base_pred, truth), value, base_pred, method,
                             truth, F, self.revealed_until_step)


# ---------- 真实曲线库加载 ----------

def load_real_instances(path: str, min_len: int = 12) -> list:
    insts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            steps, values = r.get("steps"), r.get("values")
            if not steps or not values or len(steps) < min_len:
                continue
            config = {k: r[k] for k in ("model", "task", "shot", "source") if k in r}
            iid = "_".join(str(config.get(k, "")) for k in ("source", "model", "task")) + f"_{r.get('metric','')}"
            insts.append(FCInstance(id=iid, config=config, metric=r.get("metric", "value"),
                                    steps=list(steps), values=list(values)))
    return insts


def load_all_real(dir_path: str) -> list:
    out = []
    for fn in sorted(os.listdir(dir_path)):
        if fn.endswith(".jsonl"):
            out += load_real_instances(os.path.join(dir_path, fn))
    return out


def baseline_miss(inst: FCInstance) -> float:
    """在 reveal 封顶处,最优朴素基线对真值的归一化误差。大 = 基线挂 = 判别性实例。"""
    ep = BPForecastEpisode(inst)
    ep.reveal(inst.steps[-1])
    idx = ep._idx_of(ep.revealed_until_step)
    ps, pv = inst.steps[: idx + 1], inst.values[: idx + 1]
    truth = inst.values[-1]
    _, base = strongest_baseline(ps, pv, inst.steps[-1], truth)
    rng = max(inst.values) - min(inst.values) + 1e-9
    return abs(base - truth) / rng


def load_hard_real(dir_path: str, threshold: float = 0.15) -> list:
    """效度闸门 v1:只保留朴素基线会挂的实例(平滑可外推的丢掉)。"""
    return [i for i in load_all_real(dir_path) if baseline_miss(i) >= threshold]
