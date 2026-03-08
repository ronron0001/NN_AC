#!/usr/bin/env python3
"""論文 Figure 2 再現。相対損失とスペクトル収束を可視化。"""
import sys
import os

# プロジェクトルートをパスに追加（Docker / ローカル両対応）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def _check_imports():
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # noqa: F401
    except ImportError as e:
        print("TensorFlow import failed:", e, file=sys.stderr)
        sys.exit(1)


_check_imports()

from tensorflow import keras
from src.data.dataset import generate_dataset
from src.data.kernel import get_tau_omega
from src.model.nnac import build_nnac
from src.training.callbacks import Fig2Callback, get_relative_losses


def main():
    np.random.seed(42)
    n_train, n_test = 1000, 500
    sigma = 1e-3
    _, _, d_omega = get_tau_omega()
    G_train, A_train_prob = generate_dataset(n_train, sigma, seed=0)
    G_test, A_test_prob = generate_dataset(n_test, sigma, seed=1)
    # 図2(b) 用の固定サンプル（A_prob のまま）
    G_track = G_test[:1]
    A_track = A_test_prob[0]
    # チェックポイント: 相対KLDが落ちるポイントを狙う
    checkpoint_epochs = [1, 5, 10, 20, 40, 80]
    cb = Fig2Callback(
        G_test, A_test_prob, G_track, A_track, d_omega, checkpoint_epochs
    )
    model = build_nnac()
    model.compile(optimizer="adam", loss="kl_divergence")
    model.fit(
        G_train,
        A_train_prob,
        epochs=100,
        batch_size=64,
        callbacks=[
            cb,
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=20, restore_best_weights=False
            ),
        ],
        verbose=1,
    )
    # 相対損失
    rel_kld, rel_mae, rel_rmse = get_relative_losses(cb.history)
    epochs = cb.history["epochs"]
    # 図2(a)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(epochs, rel_kld, label="KLD")
    ax.plot(epochs, rel_mae, label="MAE")
    ax.plot(epochs, rel_rmse, label="RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative loss")
    ax.legend()
    ax.set_title("(a) Relative losses")
    # 図2(b): 物理的スペクトル A(ω)（∫A dω=1）でプロット
    # 論文: A(ω) = A_prob/Δω。データセットは A_prob を返すので /d_omega で変換。
    ax = axes[1]
    _, omega, _ = get_tau_omega()
    A_plot = A_track / d_omega  # A_prob → 物理スペクトル A(ω)
    ax.plot(omega, A_plot, "k-", label="True", lw=2)
    for ep, pred_density in sorted(cb.checkpoint_preds.items()):
        ax.plot(omega, pred_density, "--", alpha=0.7, label=f"Epoch {ep}")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$A(\omega)$")
    ax.legend()
    ax.set_title("(b) Spectrum convergence")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/fig2_training_process.png", dpi=150)
    plt.close()
    print("Saved output/fig2_training_process.png")
    # 積分検証: A_prob は Σ=1、pred_density は ∫A dω = Σ A_j Δω = 1
    integral_true = np.sum(A_track)  # A_prob: Σ = 1
    last_ep = max(cb.checkpoint_preds) if cb.checkpoint_preds else 0
    integral_pred = np.sum(cb.checkpoint_preds[last_ep]) * d_omega if last_ep else 0
    print(f"Integral check: True ΣA_prob = {integral_true:.6f}, Pred ∫A dω = {integral_pred:.6f}")


if __name__ == "__main__":
    main()
