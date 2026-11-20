import numpy as np

from src.monitoring.drift import psi

rng = np.random.default_rng(0)


def test_same_distribution_has_low_psi():
    a = rng.normal(0, 1, 10_000)
    b = rng.normal(0, 1, 10_000)
    assert psi(a, b) < 0.02


def test_shifted_distribution_has_high_psi():
    a = rng.normal(0, 1, 10_000)
    b = rng.normal(1, 1, 10_000)   # mean moved by one standard deviation
    assert psi(a, b) > 0.25


def test_binary_flag_drift_is_detected():
    ref = np.array([0] * 900 + [1] * 100)   # 10% flagged in training
    cur = np.array([0] * 500 + [1] * 500)   # 50% flagged today
    assert psi(ref, cur) > 0.25


def test_missing_values_count_as_drift():
    ref = rng.normal(0, 1, 1_000)
    cur = ref.copy()
    cur[:500] = np.nan                       # half the feed went blank
    assert psi(ref, cur) > 0.25