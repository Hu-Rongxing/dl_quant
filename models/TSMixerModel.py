#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn  # 导入 nn 模块
from darts.models import TSMixerModel
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import optuna
from pathlib import Path
import logging
from dataclasses import dataclass

# 自定义设置
from config import TIMESERIES_LENGTH
from load_data.multivariate_timeseries import prepare_timeseries_data
from utils.model import loss_logger
from utils.logger import logger  # 设置日志
from models.params import get_pl_trainer_kwargs

# 设置浮点数矩阵乘法精度
torch.set_float32_matmul_precision('medium')

# 常量定义
MODEL_NAME = "TSMixerModel"
WORK_DIR = Path(f"logs/{MODEL_NAME}_logs").resolve()
MODEL_SAVE_PATH = Path(f"models/{MODEL_NAME}_best_model.pt").resolve()  # 保存模型的文件路径


@dataclass
class ModelParams:
    input_chunk_length: int
    output_chunk_length: int
    hidden_size: int
    dropout: float


# Focal Loss 实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def define_model_params(trial) -> ModelParams:
    return ModelParams(
        input_chunk_length=trial.suggest_int("input_chunk_length", 1, 64),
        output_chunk_length=trial.suggest_int("output_chunk_length", 1, 20),
        hidden_size=trial.suggest_int("hidden_size", 8, 128),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
    )


def define_model(params: ModelParams):
    model = TSMixerModel(
        input_chunk_length=params.input_chunk_length,
        output_chunk_length=params.output_chunk_length,
        hidden_size=params.hidden_size,
        dropout=params.dropout,
        loss_fn=FocalLoss(),  # 使用 Focal Loss
        optimizer_cls=torch.optim.Adam,
        pl_trainer_kwargs=get_pl_trainer_kwargs(full_training=True),
        work_dir=WORK_DIR,
        save_checkpoints=True,
        force_reset=True,
        model_name=MODEL_NAME,
        batch_size=128,
        n_epochs=50,
        random_state=42,
        log_tensorboard=False,
    )
    return model


def sigmoid_torch(x):
    return torch.sigmoid(torch.from_numpy(x)).numpy()


def train_and_evaluate(model, data):
    model.fit(
        series=data['train'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates'],
        val_series=data['val'],
        val_past_covariates=data['past_covariates'],
        val_future_covariates=data['future_covariates'],
    )

    # 清理显存
    torch.cuda.empty_cache()

    # 测试集预测
    pred_steps = TIMESERIES_LENGTH["test_length"]
    pred_input = data["test"][:-pred_steps]
    pred_series = model.predict(n=pred_steps, series=pred_input)

    # 计算精确度
    true_labels = data["test"][-pred_steps:].values()
    sigmoid_pred = sigmoid_torch(pred_series.values())
    binary_predictions = sigmoid_pred > 0.5
    precision = precision_score(true_labels.flatten().astype(int), binary_predictions.flatten().astype(int))
    logger.info(f"精确率：{precision}")

    plot_metrics(loss_logger.train_loss, loss_logger.val_loss, pred_series, data["test"], pred_steps)

    return precision


def plot_metrics(train_loss, val_loss, pred_series, test_data, pred_steps):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.plot(train_loss, label='训练损失')
    plt.plot(val_loss, label='验证损失')
    plt.legend()
    plt.show()
    plt.close()

    for stock in test_data.columns[:3]:
        plt.figure()
        test_data[-pred_steps:].data_array().sel(component=stock).plot(label=f"{stock} 实际数据")
        pred_series.data_array().sel(component=stock).plot(label=f"{stock} 预测结果")
        plt.legend()
        plt.show()
        plt.close()


def objective(trial):
    params = define_model_params(trial)
    model = define_model(params)
    precision = train_and_evaluate(model, data)
    return precision


if __name__ == '__main__':
    data = prepare_timeseries_data('training', binary=True)
    study = optuna.create_study(
        direction="maximize",
        study_name="tsmixer-model-precision-optimization",
        storage="sqlite:///data/optuna/optuna_study.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50, n_jobs=1)

    logger.info(f"最佳超参数: {study.best_params}")
    logger.info(f"最佳精确率: {study.best_value:.2%}")

    # 使用最佳参数定义模型
    best_model_params = define_model(ModelParams(**study.best_params))
    best_model = define_model(best_model_params)

    # 保存最佳模型
    best_model.save(str(MODEL_SAVE_PATH))
    logger.info(f"最佳模型已保存到: {MODEL_SAVE_PATH}")