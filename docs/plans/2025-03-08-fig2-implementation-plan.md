# 論文図2再現 実装プラン

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** arXiv:2302.11317 の Figure 2（学習過程の相対損失とスペクトル収束）を TensorFlow で再現し、Docker で再現性を確保する。

**Architecture:** モジュール分割（data / model / training）、tf.keras のみ使用、スペクトルは Softmax と Z 正規化で ΣAΔω=1 を保証。

**Tech Stack:** TensorFlow 2.15+, NumPy, Matplotlib, Docker (python:3.11-slim)

**設計ドキュメント:** `docs/plans/2025-03-08-fig2-reproduction-design.md`

---

## Task 1: プロジェクト基盤

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/model/__init__.py`, `src/training/__init__.py`
- Create: `output/.gitkeep`（出力ディレクトリ）
- Create: `tests/__init__.py`

**Step 1: requirements.txt**

```
tensorflow>=2.15.0,<2.17.0
numpy>=1.24.0,<2.0.0
matplotlib>=3.7.0
pytest>=7.0.0
```

**Step 2: ディレクトリ作成**

```bash
mkdir -p src/data src/model src/training scripts tests output
touch src/__init__.py src/data/__init__.py src/model/__init__.py src/training/__init__.py tests/__init__.py output/.gitkeep
```

**Step 3: 検証**

```bash
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected: TensorFlow version 2.15.x or 2.16.x

---

## Task 2: TensorFlow import 検証テスト

**Files:**
- Create: `tests/test_imports.py`

**Step 1: テスト作成**

```python
"""TensorFlow / tf.keras の import が通ることを確認する。"""
import pytest

def test_tensorflow_import():
    import tensorflow as tf
    assert tf is not None

def test_tf_keras_import():
    import tensorflow as tf
    from tensorflow import keras
    assert keras is not None
    assert hasattr(tf.keras, 'Model')
    assert hasattr(tf.keras.layers, 'Dense')
```

**Step 2: 実行**

```bash
pytest tests/test_imports.py -v
```

Expected: PASS

---

## Task 3: カーネルと離散化パラメータ

**Files:**
- Create: `src/data/kernel.py`

**Step 1: kernel.py 実装**

```python
"""Fermion カーネル KF(τ,ω) と G(τ) 計算。"""
import numpy as np

# 論文準拠の定数
TAU_MAX = 16.0
OMEGA_MIN, OMEGA_MAX = -15.0, 15.0
N_TAU, N_OMEGA = 512, 1024
BETA = 16.0

def get_tau_omega():
    """τ, ω の離散グリッドを返す。d_omega は可視化等に使用。"""
    tau = np.linspace(0, TAU_MAX, N_TAU, endpoint=False)
    omega = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    d_omega = (OMEGA_MAX - OMEGA_MIN) / (N_OMEGA - 1)
    return tau, omega, d_omega

def kernel_F(tau: np.ndarray, omega: np.ndarray, beta: float = BETA) -> np.ndarray:
    """KF(τ,ω) = e^(-τω) / (1 + e^(-βω)). shape: (n_tau, n_omega)"""
    # tau: (M,), omega: (N,) -> (M, N)
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
```

**Step 2: 検証**

```bash
python -c "
from src.data.kernel import get_tau_omega, kernel_F, spectrum_to_G
import numpy as np
tau, omega, d_omega = get_tau_omega()
assert tau.shape == (512,)
assert omega.shape == (1024,)
K = kernel_F(tau, omega)
assert K.shape == (512, 1024)
A = np.ones(1024) / 1024  # sum(A)=1
G = spectrum_to_G(A)
assert G.shape == (512,)
print('OK')
"
```

---

## Task 4: カーネルテスト

**Files:**
- Create: `tests/test_kernel.py`

**Step 1: テスト作成**

```python
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
```

**Step 2: 実行**

```bash
pytest tests/test_kernel.py -v
```

Expected: PASS

---

## Task 5: ASEP スペクトル生成

**Files:**
- Create: `src/data/spectra.py`

**Step 1: spectra.py 実装**

論文式(2)(3)に従う。正規化 Z で Σ A(ωⱼ)Δω = 1。

```python
"""ASEP 型スペクトル生成。正規化: Σ A(ω)Δω = 1"""
import numpy as np
from src.data.kernel import get_tau_omega

def _asep_single(omega: np.ndarray, m: float, a1: float, a2: float, b1: float, b2: float, h: float) -> np.ndarray:
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
```

**Step 2: 正規化検証**

```bash
python -c "
import numpy as np
from src.data.spectra import generate_asep_spectrum
rng = np.random.default_rng(42)
A = generate_asep_spectrum(4, rng)
assert np.isclose(np.sum(A), 1.0), f'sum={np.sum(A)}'
print('OK')
"
```

---

## Task 6: スペクトルテスト

**Files:**
- Create: `tests/test_spectra.py`

**Step 1: テスト作成**

```python
"""ASEP スペクトルの正規化を検証。"""
import numpy as np
from src.data.spectra import generate_asep_spectrum
from src.data.kernel import get_tau_omega

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
```

**Step 2: 実行**

```bash
pytest tests/test_spectra.py -v
```

Expected: PASS

---

## Task 7: データセット生成

**Files:**
- Create: `src/data/dataset.py`

**Step 1: dataset.py 実装**

```python
"""ASEP データセット生成。G にノイズ付加。"""
import numpy as np
from src.data.spectra import generate_asep_spectrum
from src.data.kernel import get_tau_omega, spectrum_to_G

def generate_dataset(n_samples: int, sigma: float, n_peaks: int = 4, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(G_noisy, A) を返す。G: (n_samples, 512), A: (n_samples, 1024). A は sum(A)=1。"""
    rng = np.random.default_rng(seed)
    A_list, G_list = [], []
    for _ in range(n_samples):
        A = generate_asep_spectrum(n_peaks, rng)
        G = spectrum_to_G(A)
        noise = rng.normal(0, sigma, size=G.shape)
        G_noisy = G + noise
        A_list.append(A)
        G_list.append(G_noisy)
    return np.array(G_list, dtype=np.float32), np.array(A_list, dtype=np.float32)
```

**Step 2: 検証**

```bash
python -c "
from src.data.dataset import generate_dataset
G, A = generate_dataset(100, 1e-3, seed=0)
assert G.shape == (100, 512)
assert A.shape == (100, 1024)
print('OK')
"
```

---

## Task 8: CNN モデル

**Files:**
- Create: `src/model/nnac.py`

**Step 1: nnac.py 実装**

tf.keras のみ使用。Softmax でスペクトル正規化。

```python
"""論文 Figure 1 の CNN。"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_nnac(n_conv: int = 8, p: int = 64, q: int = 64, n_tau: int = 512, n_omega: int = 1024) -> Model:
    inp = layers.Input(shape=(n_tau,))
    x = layers.Dense(p, activation=None)(inp)
    x = layers.Reshape((1, p))(x)
    for _ in range(n_conv):
        x = layers.Conv1D(q, 3, padding='same')(x)
        x = layers.Activation('swish')(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(n_omega)(x)
    out = layers.Softmax()(x)
    return Model(inp, out)
```

**Step 2: 検証**

```bash
python -c "
from src.model.nnac import build_nnac
m = build_nnac()
import numpy as np
x = np.random.randn(2, 512).astype(np.float32)
y = m(x)
assert y.shape == (2, 1024)
assert np.allclose(y.sum(axis=1), 1.0)
print('OK')
"
```

---

## Task 9: 図2用コールバック

**Files:**
- Create: `src/training/callbacks.py`

**Step 1: callbacks.py 実装**

各エポックでテストセットの KLD/MAE/RMSE を記録。相対損失とチェックポイント用の予測を保存。

```python
"""図2用: 相対損失の記録とチェックポイント予測の保存。"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

def kld_loss(y_true, y_pred):
    eps = 1e-10
    return -tf.reduce_sum(y_true * tf.math.log(y_pred + eps), axis=-1)

def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

class Fig2Callback(keras.callbacks.Callback):
    def __init__(self, G_test, A_test, G_track, A_track, checkpoint_epochs=None):
        super().__init__()
        self.G_test = G_test
        self.A_test = A_test
        self.G_track = G_track   # 固定サンプル (1, 512)
        self.A_track = A_track   # 真値 (1024,)
        self.checkpoint_epochs = checkpoint_epochs or []
        self.history = {'kld': [], 'mae': [], 'rmse': [], 'epochs': []}
        self.checkpoint_preds = {}  # epoch -> pred

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.G_test, verbose=0)
        kld = np.mean([kld_loss(self.A_test[i], pred[i]).numpy() for i in range(len(self.A_test))])
        mae = np.mean(np.abs(self.A_test - pred))
        mse = np.mean((self.A_test - pred) ** 2)
        rmse = np.sqrt(mse)
        self.history['kld'].append(float(kld))
        self.history['mae'].append(float(mae))
        self.history['rmse'].append(float(rmse))
        self.history['epochs'].append(epoch + 1)
        if (epoch + 1) in self.checkpoint_epochs:
            p = self.model.predict(self.G_track, verbose=0)[0]
            self.checkpoint_preds[epoch + 1] = p

def get_relative_losses(history):
    """相対損失 = loss / loss_epoch1"""
    kld0 = history['kld'][0]
    mae0 = history['mae'][0]
    rmse0 = history['rmse'][0]
    return (
        [x / kld0 for x in history['kld']],
        [x / mae0 for x in history['mae']],
        [x / rmse0 for x in history['rmse']],
    )
```

**Note:** kld_loss をバッチで扱う場合、`tf.keras.losses.kullback_leibler_divergence` も利用可。ここでは手動実装。

---

## Task 10: reproduce_fig2.py メインスクリプト

**Files:**
- Create: `scripts/reproduce_fig2.py`

**Step 1: 実装**

```python
#!/usr/bin/env python3
"""論文 Figure 2 再現。相対損失とスペクトル収束を可視化。"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def _check_imports():
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as e:
        print("TensorFlow import failed:", e, file=sys.stderr)
        sys.exit(1)

_check_imports()

import tensorflow as tf
from tensorflow import keras
from src.data.dataset import generate_dataset
from src.data.kernel import get_tau_omega
from src.model.nnac import build_nnac
from src.training.callbacks import Fig2Callback, get_relative_losses

def main():
    np.random.seed(42)
    n_train, n_test = 1000, 500
    sigma = 1e-3
    G_train, A_train = generate_dataset(n_train, sigma, seed=0)
    G_test, A_test = generate_dataset(n_test, sigma, seed=1)
    # 図2(b) 用の固定サンプル
    G_track = G_test[:1]
    A_track = A_test[0]
    # チェックポイント: 相対KLDが落ちるポイントを狙う（概ね 1,5,10,20,40 など）
    checkpoint_epochs = [1, 5, 10, 20, 40, 80]
    cb = Fig2Callback(G_test, A_test, G_track, A_track, checkpoint_epochs)
    model = build_nnac()
    model.compile(optimizer='adam', loss='kullback_leibler_divergence')
    model.fit(G_train, A_train, epochs=100, batch_size=64, callbacks=[
        cb,
        keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
    ], verbose=1)
    # 相対損失
    rel_kld, rel_mae, rel_rmse = get_relative_losses(cb.history)
    epochs = cb.history['epochs']
    # 図2(a)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(epochs, rel_kld, label='KLD')
    ax.plot(epochs, rel_mae, label='MAE')
    ax.plot(epochs, rel_rmse, label='RMSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative loss')
    ax.legend()
    ax.set_title('(a) Relative losses')
    # 図2(b)
    ax = axes[1]
    tau, omega, _ = get_tau_omega()
    ax.plot(omega, A_track, 'k-', label='True', lw=2)
    for ep, pred in sorted(cb.checkpoint_preds.items()):
        ax.plot(omega, pred, '--', alpha=0.7, label=f'Epoch {ep}')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$A(\omega)$')
    ax.legend()
    ax.set_title('(b) Spectrum convergence')
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/fig2_training_process.png', dpi=150)
    plt.close()
    print("Saved output/fig2_training_process.png")

if __name__ == '__main__':
    main()
```

**Note:** `keras.callbacks.EarlyStopping` は `from tensorflow import keras` が必要。スクリプト冒頭で import。

**Step 2: 修正（keras の import）**

```python
from tensorflow import keras
# ...
model.fit(..., callbacks=[cb, keras.callbacks.EarlyStopping(...)])
```

---

## Task 11: Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "scripts/reproduce_fig2.py"]
```

**Step 2: ビルド・実行**

```bash
docker build -t nnac-fig2 .
docker run -v $(pwd)/output:/app/output nnac-fig2
```

Expected: `output/fig2_training_process.png` が生成される。

---

## Task 12: 統合検証

**Step 1: ローカル実行**

```bash
cd /Users/matsumototakuya/NN_AC
pip install -r requirements.txt
python scripts/reproduce_fig2.py
```

**Step 2: Docker 実行**

```bash
docker build -t nnac-fig2 .
docker run -v $(pwd)/output:/app/output nnac-fig2
```

**Step 3: 全テスト**

```bash
pytest tests/ -v
```

Expected: 全 PASS、図2画像が output に出力される。
