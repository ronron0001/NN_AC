#!/usr/bin/env python3
"""スペクトル正規化の検証（TensorFlow不要）
データセットは A_prob（Σ=1）を返す。∫A dω = Σ A_prob = 1。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.kernel import get_tau_omega
from src.data.dataset import generate_dataset

_, _, d_omega = get_tau_omega()
_, A_prob = generate_dataset(10, 1e-3, seed=42)

print(f"d_omega = {d_omega}")
for i in range(3):
    integral = A_prob[i].sum()  # A_prob: Σ = 1 = ∫A dω（論文の Δω 吸収形式）
    print(f"  サンプル {i}: Σ A_prob = ∫A dω = {integral:.10f}  (期待値: 1.0)")
