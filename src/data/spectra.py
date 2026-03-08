"""ASEP 型スペクトル生成。正規化: sum(A)=1（論文の Δω 吸収形式）"""
import numpy as np
from src.data.kernel import get_tau_omega


def _asep_single(
    omega: np.ndarray,
    m: float,
    a1: float,
    a2: float,
    b1: float,
    b2: float,
    h: float,
) -> np.ndarray:
    """単一 ASEP ピーク。式(3)"""
    out = np.zeros_like(omega, dtype=float)
    left = omega < m
    right = omega >= m
    out[left] = h * np.exp(-np.power(np.abs(m - omega[left]) / a1, b1))
    out[right] = h * np.exp(-np.power(np.abs(omega[right] - m) / a2, b2))
    return out


def generate_asep_spectrum(n_peaks: int, rng: np.random.Generator) -> np.ndarray:
    """1〜n_peaks 個の ASEP ピークを重ね、sum(A)=1 に正規化（論文の Δω 吸収形式）。"""
    _, omega, _ = get_tau_omega()
    A = np.zeros_like(omega)
    n = rng.integers(1, n_peaks + 1) if n_peaks > 1 else 1
    for _ in range(n):
        m = rng.uniform(-5, 5)
        a1, a2 = rng.uniform(0.3, 3, 2)
        b1, b2 = rng.uniform(1, 3, 2)
        h = rng.uniform(0.2, 1)
        A += _asep_single(omega, m, a1, a2, b1, b2, h)
    A = np.clip(A, 1e-20, None)  # 数値安定
    Z = np.sum(A)
    if Z <= 0:
        return generate_asep_spectrum(n_peaks, rng)  # 再試行
    return A / Z  # sum(A)=1
