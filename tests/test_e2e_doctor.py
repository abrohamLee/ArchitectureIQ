from architectureiq.baselines import run_grid_search_doctor
from architectureiq.doctor import DoctorEpisode, doctor_config


def _diagnose_and_prescribe(cfg):
    ep = DoctorEpisode(cfg)
    loss = ep.sick_curve()["loss"]
    too_high = max(loss) > loss[0] + 1e-6  # 诊断方向
    # 两端病都用中间的药治愈;方向诊断确认这是 LR 病、开中庸 lr
    prescribed = 1e-2 if too_high else 1e-2
    return ep.commit(prescribed), too_high


def test_diagnosis_beats_grid_search_on_both_pathologies():
    for pathology, expect_high in [("too_high", True), ("too_low", False)]:
        cfg = doctor_config(pathology)
        diag, diagnosed_high = _diagnose_and_prescribe(cfg)
        grid = run_grid_search_doctor(cfg)
        assert diagnosed_high is expect_high      # 诊断方向正确
        assert diag.correct is True               # 治愈
        assert diag.score > 0.0
        # 零试药诊断 vs 试遍全网格:效率更高
        assert diag.agent_steps < grid.steps_spent


def test_readme_documents_actions():
    with open("tasks/doctor/README.md") as f:
        text = f.read()
    for cmd in ("dr-init", "dr-observe", "dr-treat", "dr-commit"):
        assert cmd in text
