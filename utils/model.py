import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 确保真实值不为零，以避免除零错误
        epsilon = 1e-10
        mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        return mape


class WeightedMSELoss(nn.Module):
    """
    加权均方误差损失，对于较大数值的预测结果，给予更大的权重。
    对于噪声较多的模型，让模型更多关注更大的值或更小的值，防止预测结果过度向平均值收敛。
    """
    def __init__(self, power=1):
        super(WeightedMSELoss, self).__init__()
        self.power = power

    def forward(self, y_pred, y_true):
        # 根据真实值计算权重，较大值权重大
        weights = y_true ** self.power
        loss = weights * (y_pred - y_true) ** 2
        return torch.mean(loss)


class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))


loss_logger = LossLogger()