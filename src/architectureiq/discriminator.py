import math

from architectureiq.trainer import LearningCurve


def curve_features(curve: LearningCurve) -> list[float]:
    return list(curve.loss) + list(curve.acc)


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _min_dist(feat: list[float], vecs: list[list[float]]) -> float:
    return min(_dist(feat, v) for v in vecs)


def discriminate_signatures(
    reference: dict[str, list[list[float]]],
    queries: list[tuple[str, list[float]]],
) -> tuple[float, float]:
    """在定长特征向量(签名)上做最近邻判别。

    reference: {arch: [签名向量, ...]};queries: [(true_arch, 签名向量), ...]。
    返回 (accuracy, margin),margin = 平均(次近类距离 − 预测类距离)/ 参考类间尺度。
    """
    archs = list(reference.keys())
    correct = 0
    margins: list[float] = []
    scale = 1e-9

    # 参考集内的最大类间最近距离,做归一化尺度
    for i, a in enumerate(archs):
        for b in archs[i + 1 :]:
            for v in reference[a]:
                scale = max(scale, _min_dist(v, reference[b]))

    for true_arch, feat in queries:
        dists = {arch: _min_dist(feat, reference[arch]) for arch in archs}
        pred = min(dists, key=dists.get)
        if pred == true_arch:
            correct += 1
        ordered = sorted(dists.values())
        gap = (ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
        margins.append(gap / scale)

    acc = correct / len(queries)
    margin = sum(margins) / len(margins)
    return acc, margin


def discriminate(
    reference: dict[str, list[LearningCurve]],
    queries: list[tuple[str, LearningCurve]],
) -> tuple[float, float]:
    """在 learning curve 上判别(经 curve_features 转成签名向量后委托)。"""
    ref_vecs = {arch: [curve_features(c) for c in cs] for arch, cs in reference.items()}
    query_vecs = [(arch, curve_features(c)) for arch, c in queries]
    return discriminate_signatures(ref_vecs, query_vecs)
