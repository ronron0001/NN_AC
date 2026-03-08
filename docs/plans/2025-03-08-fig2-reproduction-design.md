# 論文図2再現 設計ドキュメント

> arXiv:2302.11317 "Neural Network Analytic Continuation for Monte Carlo: Improvement by Statistical Errors"

**日付**: 2025-03-08

## 1. 概要

論文の Figure 2（学習過程の追跡とスペクトル収束の可視化）を TensorFlow/Keras で再現する。Docker（CPU）で再現性を確保し、スペクトルは正規化（ΣA(ω)Δω=1）を保証する。

## 2. アーキテクチャとディレクトリ構成

```
NN_AC/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── spectra.py
│   │   ├── kernel.py
│   │   └── dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── nnac.py
│   └── training/
│       ├── __init__.py
│       └── callbacks.py
├── scripts/
│   └── reproduce_fig2.py
├── tests/
│   ├── test_imports.py
│   ├── test_spectra.py
│   └── test_kernel.py
└── docs/plans/
```

## 3. データパイプライン

### 3.1 パラメータ
- τ: [0, 16], 512点
- ω: [-15, 15], 1024点, Δω = 30/1024
- β: 16
- ノイズ: σ = 10⁻³

### 3.2 ASEP スペクトル
- 式(2)(3)に従い 1〜4 ピークをランダム生成
- Z で正規化し Σ A(ωⱼ)Δω = 1

### 3.3 カーネル
- KF(τ,ω) = e^(-τω) / (1 + e^(-βω))
- G(τᵢ) = Σⱼ K(τᵢ,ωⱼ) A(ωⱼ) Δω

### 3.4 スペクトル正規化
- モデル出力: Softmax で Σ A_pred = 1
- 教師データ: Z で Σ A_true Δω = 1

## 4. CNN モデル

- Dense(512→64) → Reshape → Conv1d+Swish×8 → AvgPool → Dense → Softmax
- n=8, p=64, q=64
- 損失: KLD

## 5. 学習と図2記録

- 1000 サンプル/エポック（可視化用）
- 相対損失 = loss_epoch / loss_epoch1
- チェックポイントで固定サンプルの予測スペクトルを保存

## 6. TensorFlow import 方針

- `import tensorflow as tf` と `tf.keras` のみ使用
- 単体 `keras` パッケージは使用しない
- tensorflow>=2.15,<2.17

## 7. Docker

- ベース: python:3.11-slim
- CPU のみ
- 出力: output/ にマウント

## 8. エラーハンドリング・テスト

- import 失敗時の明示的メッセージ
- tests: imports, spectra 正規化, kernel 形状
