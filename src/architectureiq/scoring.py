def rhae_ml_score(correct: bool, agent_steps: int, human_steps: int) -> float:
    """RHAE-ML:正确性闸门 × 平方效率。

    correct=False -> 0(把「难≠可区分,瞎猜得零」焊进评分)。
    correct=True  -> min(1, human_steps / agent_steps)^2(平方狠惩罚低效探索)。
    """
    if not correct:
        return 0.0
    efficiency = min(1.0, human_steps / max(agent_steps, 1))
    return efficiency ** 2


def brute_force_ratio(agent_steps: int, bruteforce_steps: int) -> float:
    """agent 花费相对暴力基线的比;<1 = 比暴力省。全自动、无需真人。"""
    return agent_steps / max(bruteforce_steps, 1)
