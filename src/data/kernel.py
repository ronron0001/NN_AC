"""Fermion カーネル KF(τ,ω) と G(τ) 計算。"""
import numpy as np

# 論文準拠の定数
TAU_MAX = 16.0
OMEGA_MIN, OMEGA_MAX = -15.0, 15.0
N_TAU, N_OMEGA = 512, 1024
BETA = 16.0


def get_tau_omega():
    """τ, ω の離散グリッドを返す。d_omega は設計ドキュメント準拠で Δω = 30/1024。"""
    tau = np.linspace(0, TAU_MAX, N_TAU, endpoint=False)
    omega = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    # 設計ドキュメント: Δω = 30/1024（セル幅）。∫A dω = Σ A_j Δω = 1 の離散化に使用
    d_omega = (OMEGA_MAX - OMEGA_MIN) / N_OMEGA
    return tau, omega, d_omega


def kernel_F(tau: np.ndarray, omega: np.ndarray, beta: float = BETA) -> np.ndarray:
    """KF(τ,ω) = e^(-τω) / (1 + e^(-βω)). shape: (n_tau, n_omega)"""
    tau = np.asarray(tau)[:, np.newaxis]
    omega = np.asarray(omega)[np.newaxis, :]
    exp_tw = np.exp(-tau * omega)
    denom = 1.0 + np.exp(-beta * omega)
    return exp_tw / denom


def spectrum_to_G(A: np.ndarray) -> np.ndarray:
    """G(τᵢ) = Σⱼ K(τᵢ,ωⱼ) A(ωⱼ). 論文では Δω を A に吸収。A は sum(A)=1 の確率分布。"""
    tau, omega, _ = get_tau_omega()
    K = kernel_F(tau, omega)
    return K @ A
