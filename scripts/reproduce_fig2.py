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
    G_train, A_train_density = generate_dataset(n_train, sigma, seed=0)
    G_test, A_test_density = generate_dataset(n_test, sigma, seed=1)
    # 学習用: モデルは確率 (sum=1) を出力するので、ターゲットも確率に
    A_train_prob = A_train_density * d_omega
    # 図2(b) 用の固定サンプル（密度のまま）
    G_track = G_test[:1]
    A_track = A_test_density[0]
    # チェックポイント: 相対KLDが落ちるポイントを狙う
    checkpoint_epochs = [1, 5, 10, 20, 40, 80]
    cb = Fig2Callback(
        G_test, A_test_density, G_track, A_track, d_omega, checkpoint_epochs
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
    # 図2(b): スペクトル密度 A(ω)/Δω をプロット（∫A dω = 1 になる正規化）
    ax = axes[1]
    _, omega, d_omega = get_tau_omega()
    A_track_density = A_track / d_omega  # 確率→密度: ∫A dω = Σ(A/Δω)·Δω = 1
    ax.plot(omega, A_track_density, "k-", label="True", lw=2)
    for ep, pred in sorted(cb.checkpoint_preds.items()):
        ax.plot(omega, pred / d_omega, "--", alpha=0.7, label=f"Epoch {ep}")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$A(\omega)$")
    ax.legend()
    ax.set_title("(b) Spectrum convergence")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/fig2_training_process.png", dpi=150)
    plt.close()
    print("Saved output/fig2_training_process.png")


if __name__ == "__main__":
    main()
