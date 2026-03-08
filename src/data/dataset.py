"""ASEP データセット生成。G にノイズ付加。"""
import numpy as np
from src.data.spectra import generate_asep_spectrum
from src.data.kernel import get_tau_omega, spectrum_to_G


def generate_dataset(
    n_samples: int,
    sigma: float,
    n_peaks: int = 4,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(G_noisy, A_density) を返す。G: (n_samples, 512), A_density: (n_samples, 1024).
    A_density は ∫A dω = 1 を満たすスペクトル密度（正規化済み）。
    """
    _, _, d_omega = get_tau_omega()
    rng = np.random.default_rng(seed)
    A_list, G_list = [], []
    for _ in range(n_samples):
        A_prob = generate_asep_spectrum(n_peaks, rng)  # sum(A_prob)=1
        G = spectrum_to_G(A_prob)
        noise = rng.normal(0, sigma, size=G.shape)
        G_noisy = G + noise
        A_density = A_prob / d_omega  # ∫A_density dω = 1
        A_list.append(A_density)
        G_list.append(G_noisy)
    return np.array(G_list, dtype=np.float32), np.array(A_list, dtype=np.float32)
