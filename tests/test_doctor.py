from architectureiq.doctor import DoctorEpisode, doctor_config


def test_sick_curve_too_high_shows_loss_spike_above_start():
    ep = DoctorEpisode(doctor_config("too_high"))
    loss = ep.sick_curve()["loss"]
    assert max(loss) > loss[0]  # LR 太高的签名:loss 曾冲高过起点


def test_sick_curve_too_low_no_loss_spike():
    ep = DoctorEpisode(doctor_config("too_low"))
    loss = ep.sick_curve()["loss"]
    assert max(loss) <= loss[0] + 1e-6  # 太低:loss 单调不增于起点(平台)


def test_commit_cure_lr_correct_and_scores_positive():
    ep = DoctorEpisode(doctor_config("too_high"))
    res = ep.commit(1e-2)  # 中间的药治愈
    assert res.correct is True and res.score > 0.0


def test_commit_sick_lr_incorrect_zero_score():
    cfg = doctor_config("too_high")
    ep = DoctorEpisode(cfg)
    res = ep.commit(cfg.sick_lr)  # 还开原病药 -> 不治
    assert res.correct is False and res.score == 0.0


def test_treat_deducts_budget_and_reports_cure():
    ep = DoctorEpisode(doctor_config("too_low"))
    r = ep.treat(1e-2)
    assert r.cured is True and r.cost == ep.treat_cost()
    assert r.budget_remaining == ep.config.budget_steps - r.cost


def test_zero_treatment_correct_commit_is_max_efficiency():
    # 凭诊断零试药直接开对药 -> 效率满分
    ep = DoctorEpisode(doctor_config("too_high"))
    res = ep.commit(1e-2)
    assert res.agent_steps == 0 and res.score == 1.0
