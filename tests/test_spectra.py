"""ASEP スペクトルの正規化を検証。"""
import numpy as np
from src.data.spectra import generate_asep_spectrum


def test_asep_normalization():
    rng = np.random.default_rng(123)
    for _ in range(10):
        A = generate_asep_spectrum(4, rng)
        assert np.isclose(np.sum(A), 1.0), f"sum={np.sum(A)}"


def test_asep_non_negative():
    rng = np.random.default_rng(456)
    A = generate_asep_spectrum(4, rng)
    assert np.all(A >= 0)
    assert A.shape == (1024,)
