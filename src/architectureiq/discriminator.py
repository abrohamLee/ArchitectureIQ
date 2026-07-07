import math

from architectureiq.trainer import LearningCurve


def curve_features(curve: LearningCurve) -> list[float]:
    return list(curve.loss) + list(curve.acc)


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _min_dist_to_arch(feat: list[float], curves: list[LearningCurve]) -> float:
    return min(_dist(feat, curve_features(c)) for c in curves)


def discriminate(
    reference: dict[str, list[LearningCurve]],
    queries: list[tuple[str, LearningCurve]],
) -> tuple[float, float]:
    archs = list(reference.keys())
    correct = 0
    margins: list[float] = []
    scale = 1e-9

    # 用参考集内的平均类间最近距离做归一化尺度
    for i, a in enumerate(archs):
        for b in archs[i + 1 :]:
            for c in reference[a]:
                scale = max(scale, _min_dist_to_arch(curve_features(c), reference[b]))

    for true_arch, curve in queries:
        feat = curve_features(curve)
        dists = {arch: _min_dist_to_arch(feat, reference[arch]) for arch in archs}
        pred = min(dists, key=dists.get)
        if pred == true_arch:
            correct += 1
        ordered = sorted(dists.values())
        gap = (ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
        margins.append(gap / scale)

    acc = correct / len(queries)
    margin = sum(margins) / len(margins)
    return acc, margin
