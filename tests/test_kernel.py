"""kernel の形状と値の妥当性を検証。"""
import numpy as np
from src.data.kernel import get_tau_omega, kernel_F, spectrum_to_G


def test_tau_omega_shapes():
    tau, omega, d_omega = get_tau_omega()
    assert tau.shape == (512,)
    assert omega.shape == (1024,)
    assert 0 <= tau[0] and tau[-1] < 16
    assert omega[0] == -15 and omega[-1] == 15


def test_kernel_shape():
    tau, omega, _ = get_tau_omega()
    K = kernel_F(tau, omega)
    assert K.shape == (512, 1024)
    assert np.all(K >= 0)
    assert np.all(np.isfinite(K))


def test_spectrum_to_G_shape():
    tau, omega, d_omega = get_tau_omega()
    A = np.ones(1024) / 1024  # sum(A)=1（論文の吸収形式）
    G = spectrum_to_G(A)
    assert G.shape == (512,)
    assert np.all(np.isfinite(G))
