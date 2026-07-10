from architectureiq.pretrainbench.gate import pair_gate


def test_clear_separation_passes():
    r = pair_gate([3.0, 3.02, 2.98], [3.5, 3.52, 3.48])
    assert r.passed and r.better == "a" and r.win_rate == 1.0 and r.gap > 0.1


def test_overlap_fails():
    r = pair_gate([3.0, 3.4, 3.2], [3.1, 3.3, 3.25])
    assert not r.passed


def test_tiny_gap_fails_even_if_consistent():
    r = pair_gate([3.000, 3.001, 3.002], [3.010, 3.011, 3.012], min_gap=0.05)
    assert r.win_rate == 1.0 and not r.passed
