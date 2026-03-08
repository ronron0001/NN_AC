# NN_AC: 論文図2再現

arXiv:2302.11317「Neural Network Analytic Continuation for Monte Carlo: Improvement by Statistical Errors」の Figure 2 を再現する実装です。

## 実行方法（推奨: Docker）

```bash
docker build -t nnac-fig2 .
docker run -v $(pwd)/output:/app/output nnac-fig2
```

生成された図は `output/fig2_training_process.png` に保存されます。

## ローカル実行

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/reproduce_fig2.py
```

**注意**: Mac（特に Apple Silicon）では TensorFlow/NumPy の互換性問題でローカル実行が失敗する場合があります。その場合は Docker を使用してください。

## テスト

```bash
pytest tests/test_kernel.py tests/test_spectra.py -v
```

`test_imports.py` は TensorFlow を import するため、上記環境問題がある場合はスキップされます。

## スペクトル正規化

- 教師データ・予測スペクトルは **sum(A)=1**（論文の Δω 吸収形式）で正規化されています
- モデル出力層は Softmax により自動的に正規化されます
