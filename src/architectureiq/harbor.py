"""Harbor verifier-side 独立评分。

Harbor 模型:agent 在容器里用 CLI 玩任务,把最终答案写到 /workspace/submission.json;
评分时 Harbor 挂载隐藏的 /tests/(agent 看不到),verifier 用**隐藏答案独立重算**分数
(不信 agent 自报),写进 reward.txt。这里就是"独立重算"那一步——纯 stdlib + 本包,可本地单测。

reward ∈ [0,1]。v1 Harbor 先用**正确性**(verifier 可独立验证、不可作弊);效率加权需要
可信的预算追踪(Harbor 原生不给),列为后续。
"""
from architectureiq.datasets import DatasetSpec


def score_submission(task_id: str, hidden: dict, submission: dict) -> float:
    """按 (隐藏答案, agent 提交) 独立算 reward ∈ [0,1]。"""
    if task_id == "fingerprint_lever":
        # 独立重算 agent 设计的数据集能否分开杠杆(margin 过阈值)
        from architectureiq.leverfingerprint import score_lever_fingerprint
        spec = DatasetSpec(
            family=submission["family"],
            n_samples=int(submission.get("n_samples", 300)),
            modulus=int(submission.get("modulus", 7)),
            n_bits=int(submission.get("n_bits", 8)),
            label_noise=float(submission.get("label_noise", 0.0)),
        )
        fp = score_lever_fingerprint(
            hidden["lever_family"], spec,
            hidden["ref_seeds"], hidden["query_seeds"], hidden["probe_steps"],
        )
        return 1.0 if fp.margin >= hidden["correct_margin"] else 0.0

    if task_id == "blackbox_lever":
        return 1.0 if submission.get("guess") == hidden["hidden_value"] else 0.0

    if task_id == "diagnostic":
        return 1.0 if submission.get("cause") == hidden["pathology"] else 0.0

    if task_id == "windtunnel":
        from architectureiq.windtunnel import WindTunnel, default_windtunnel_config
        wt = WindTunnel(default_windtunnel_config(hidden["seed"]))
        chosen = submission.get("candidate")
        if chosen not in wt.large:
            return 0.0
        regret = max(wt.large.values()) - wt.large[chosen]
        return 1.0 if regret <= hidden["regret_threshold"] else 0.0

    raise ValueError(f"unknown Harbor task_id: {task_id}")
