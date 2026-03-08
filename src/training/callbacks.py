"""図2用: 相対損失の記録とチェックポイント予測の保存。"""
import numpy as np
from tensorflow import keras


class Fig2Callback(keras.callbacks.Callback):
    def __init__(self, G_test, A_test, G_track, A_track, d_omega, checkpoint_epochs=None):
        super().__init__()
        self.G_test = G_test
        self.A_test = A_test  # A_prob (Σ=1、論文の Δω 吸収形式)
        self.G_track = G_track
        self.A_track = A_track  # A_prob
        self.d_omega = d_omega
        self.checkpoint_epochs = set(checkpoint_epochs or [])
        self.history = {"kld": [], "mae": [], "rmse": [], "epochs": []}
        self.checkpoint_preds = {}  # epoch -> pred_density (∫A dω=1)

    def on_epoch_end(self, epoch, logs=None):
        pred_prob = self.model.predict(self.G_test, verbose=0)  # 確率 (sum=1)
        pred_density = pred_prob / self.d_omega  # 密度 A(ω) = A_prob/Δω
        A_density = self.A_test / self.d_omega  # A_prob → 密度
        eps = 1e-10
        # 論文式(6): D_KL = Σ A_true ln(A_true/A_pred) = CE - H(A_true)
        # 密度形式で計算
        kld_list = []
        for i in range(len(self.A_test)):
            ce = -np.sum(A_density[i] * self.d_omega * np.log(pred_density[i] + eps))
            h = -np.sum(A_density[i] * self.d_omega * np.log(A_density[i] + eps))
            kld_list.append(ce - h)  # KL = CE - H
        kld = np.mean(kld_list)
        mae = np.mean(np.abs(A_density - pred_density))
        mse = np.mean((A_density - pred_density) ** 2)
        rmse = np.sqrt(mse)
        self.history["kld"].append(float(kld))
        self.history["mae"].append(float(mae))
        self.history["rmse"].append(float(rmse))
        self.history["epochs"].append(epoch + 1)
        if (epoch + 1) in self.checkpoint_epochs:
            p_prob = self.model.predict(self.G_track, verbose=0)[0]
            self.checkpoint_preds[epoch + 1] = p_prob / self.d_omega  # 密度で保存


def get_relative_losses(history):
    """相対損失 = loss / loss_epoch1"""
    kld0 = history["kld"][0]
    mae0 = history["mae"][0]
    rmse0 = history["rmse"][0]
    return (
        [x / kld0 for x in history["kld"]],
        [x / mae0 for x in history["mae"]],
        [x / rmse0 for x in history["rmse"]],
    )
