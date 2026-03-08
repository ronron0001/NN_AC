"""論文 Figure 1 の CNN。"""
from tensorflow.keras import Model, layers


def build_nnac(
    n_conv: int = 8,
    p: int = 64,
    q: int = 64,
    n_tau: int = 512,
    n_omega: int = 1024,
) -> Model:
    inp = layers.Input(shape=(n_tau,))
    x = layers.Dense(p, activation=None)(inp)
    x = layers.Reshape((p, 1))(x)  # (batch, p, 1): p time steps, 1 channel
    for _ in range(n_conv):
        x = layers.Conv1D(q, 3, padding="same")(x)
        x = layers.Activation("swish")(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(n_omega)(x)
    out = layers.Softmax()(x)
    return Model(inp, out)
