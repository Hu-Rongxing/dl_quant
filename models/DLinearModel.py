#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import BCEWithLogitsLoss
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.callbacks import TFMProgressBar
from darts.models import DLinearModel  # Import the new model
from sklearn.metrics import precision_score
from pathlib import Path
import matplotlib.pyplot as plt
import optuna

# 自定义设置
from config import TIMESERIES_LENGTH
from load_data.multivariate_timeseries import prepare_timeseries_data
from utils.model import LossLogger
from models.params import get_pl_trainer_kwargs

# 设置浮点数矩阵乘法精度
torch.set_float32_matmul_precision('medium')

# 初始化损失记录器
loss_logger = LossLogger()

# 初始化进度条
progress_bar = TFMProgressBar(enable_sanity_check_bar=False, enable_validation_bar=False)

# 设置早停策略
early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-6,
    mode="min",
)

model_name = "DLinearModel"  # Update model name
work_dir = Path(f"logs/{model_name}_logs").resolve()
study_storage="sqlite:///data/optuna/optuna_study.db"

# 准备训练和验证数据
data = prepare_timeseries_data('training')

# 初始化DLinearModel
model = DLinearModel(
    input_chunk_length=1,
    output_chunk_length=1,
    pl_trainer_kwargs=get_pl_trainer_kwargs(),
    work_dir=work_dir,
    save_checkpoints=True,
    force_reset=True,
    model_name=model_name
)

# 训练模型
model.fit(
    series=data['train'],
    past_covariates=data['past_covariates'],
    future_covariates=data['future_covariates'],
    val_series=data['val'],
    val_past_covariates=data['past_covariates'],
    val_future_covariates=data['future_covariates'],
)

# 从检查点加载模型
model = model.load_from_checkpoint(model_name=model_name, work_dir=work_dir)

# 预测步骤
pred_steps = TIMESERIES_LENGTH["test_length"]
pred_input = data["test"][:-pred_steps]

# 进行预测
pred_series = model.predict(n=pred_steps, series=pred_input)

# 对预测结果进行二值化和展平
true_labels = data["test"][-pred_steps:].values() > 0.5
true_labels = true_labels.astype(int).flatten()
binary_predictions = pred_series.values() > 0.5
binary_predictions = binary_predictions.astype(int).flatten()

# 绘制训练和验证损失
plt.plot(loss_logger.train_loss, label='训练损失')
plt.plot(loss_logger.val_loss, label='验证损失')
plt.legend()
plt.show()

# 计算精确率
precision = precision_score(true_labels, binary_predictions)
print(f"预测精确率：{precision}")

# 绘制实际数据和预测结果（前3个股票）
for i, stock in enumerate(data["test"].columns[:3]):
    plt.figure()  # 为每个股票创建一个新图
    data["test"][-pred_steps:].data_array().sel(component=stock).plot(label=f"{stock} 实际数据")
    pred_series.data_array().sel(component=stock).plot(label=f"{stock} 预测结果")
    plt.legend()
    plt.show()

# 定义超参数优化目标函数
def objective(trial):
    input_chunk_length = trial.suggest_int("input_chunk_length", 1, 128)
    output_chunk_length = trial.suggest_int("output_chunk_length", 1, 20)

    model = DLinearModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        pl_trainer_kwargs=get_pl_trainer_kwargs(),
        work_dir=work_dir,
        save_checkpoints=True,
        force_reset=True,
        model_name=model_name
    )

    # 训练模型
    model.fit(
        series=data['train'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates'],
        val_series=data['val'],
        val_past_covariates=data['past_covariates'],
        val_future_covariates=data['future_covariates'],
    )

    # 从检查点加载模型
    # model = model.load_from_checkpoint(model_name=model_name, work_dir=work_dir)

    # 预测步骤
    pred_steps = TIMESERIES_LENGTH["test_length"]
    pred_input = data["test"][:-pred_steps]

    # 进行预测
    pred_series = model.predict(n=pred_steps, series=pred_input)

    # 对预测结果进行二值化和展平
    true_labels = data["test"][-pred_steps:].values() > 0.5
    true_labels = true_labels.astype(int).flatten()
    binary_predictions = pred_series.values() > 0.5
    binary_predictions = binary_predictions.astype(int).flatten()

    # 计算精确率
    precision = precision_score(true_labels, binary_predictions)
    print(f"精确率：{precision}")

    return precision  # 返回精确度作为目标

def delete_study(study_name, storage_url):
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"成功删除 study：{study_name}")
    except KeyError:
        print(f"找不到 study：{study_name}")

if __name__ == '__main__':

    delete_study(f"{model_name}-optimization", "sqlite:///data/optuna/optuna_study.db")

    # 创建一个新的Optuna研究并开始优化
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}-optimization",
        storage=study_storage,  # 指定SQLite数据库存储位置
        load_if_exists=True  # 如果已存在则加载研究
    )  # 我们想最大化精确度
    study.optimize(objective, n_trials=50)  # 这里可以调整试验次数

    print("最佳超参数: ", study.best_params)
    print("最佳精确率: ", study.best_value)