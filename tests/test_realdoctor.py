from architectureiq.realcurvebank import get_curve
from architectureiq.realdoctor import (
    RealDoctorConfig,
    RealDoctorEpisode,
    is_pathological,
    real_doctor_ids,
)


def _pick():
    ids = real_doctor_ids()
    sick = [i for i in ids if is_pathological(get_curve(i).acc)]
    healthy = [i for i in ids if not is_pathological(get_curve(i).acc)]

    def early(i):
        v = get_curve(i).acc
        return v.index(min(v)) / len(v) < 0.3

    early_sick = [i for i in sick if early(i)]
    return early_sick[0], healthy[0], sick


def test_pool_has_sick_and_healthy_real_curves():
    ids = real_doctor_ids()
    _, _, sick = _pick()
    assert len(ids) >= 50
    assert len(sick) >= 5  # 天然发散的 run


def test_early_detection_beats_majority_baseline():
    early_sick, healthy_id, _ = _pick()
    # 诊断者:从开局前缀就抓出发散(0 reveal)
    r = RealDoctorEpisode(RealDoctorConfig(curve_id=early_sick)).diagnose("pathological")
    assert r.correct is True and r.score == 1.0
    # 语义盲地板:总猜 healthy -> 在病态曲线上必错
    b = RealDoctorEpisode(RealDoctorConfig(curve_id=early_sick)).diagnose("healthy")
    assert b.correct is False and b.score == 0.0
    # 健康曲线正确判为 healthy
    h = RealDoctorEpisode(RealDoctorConfig(curve_id=healthy_id)).diagnose("healthy")
    assert h.correct is True and h.score == 1.0


def test_more_reveals_cost_budget_and_lower_efficiency():
    _, healthy_id, _ = _pick()
    ep = RealDoctorEpisode(RealDoctorConfig(curve_id=healthy_id))
    ep.reveal(); ep.reveal(); ep.reveal()  # 30 > human_budget 20
    r = ep.diagnose("healthy")
    assert r.correct is True and 0.0 < r.score < 1.0  # 揭得多 -> 效率被平方惩罚
