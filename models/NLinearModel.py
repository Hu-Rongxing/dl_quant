#!/usr/bin/env python
# coding: utf-8

import torch
from torch.nn import MSELoss
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.callbacks import TFMProgressBar
from darts.models import NLinearModel  # Import the new model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, classification_report
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import optuna
from config import TIMESERIES_LENGTH
from load_data.multivariate_timeseries import prepare_timeseries_data
from utils.model import LossLogger
from models.params import get_pl_trainer_kwargs

# 设置矩阵乘法的浮点精度
torch.set_float32_matmul_precision('medium')

# 初始化损失记录器和进度条
loss_logger = LossLogger()
progress_bar = TFMProgressBar(enable_sanity_check_bar=False, enable_validation_bar=False)

# 早停策略
early_stopper = EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=1e-6,
    mode="min",
)

model_name = "NLinearModel"
work_dir = Path(f"logs/{model_name}_logs").resolve()

# 准备训练和验证数据
data = prepare_timeseries_data('training')

# 初始化 NLinearModel
model = NLinearModel(
    input_chunk_length=15,
    output_chunk_length=1,
    pl_trainer_kwargs=get_pl_trainer_kwargs(),
    work_dir=work_dir,
    save_checkpoints=True,
    force_reset=True,
    model_name=model_name,
    # loss_fn=MSELoss()  # 使用均方误差损失函数（可选，如果模型默认已经是回归损失）
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

# 获取预测值和真实值
predicted_values = pred_series.values()
true_values = data["test"][-pred_steps:].values()

# 检查预测值和真实值的形状
print("真实值形状:", true_values.shape)  # 输出真实值的形状
print("预测值形状:", predicted_values.shape)  # 输出预测值的形状

# 计算回归指标
mse = mean_squared_error(true_values, predicted_values)
mae = mean_absolute_error(true_values, predicted_values)
r2 = r2_score(true_values, predicted_values)
print(f"均方误差 (MSE)：{mse:.4f}")
print(f"均绝对误差 (MAE)：{mae:.4f}")
print(f"R² 分数：{r2:.4f}")

# 打印更详细的评估报告（可选）
print(f"详细评估报告:\nMSE: {mse}\nMAE: {mae}\nR²: {r2}")

# **新增部分：计算预测结果大于0时的精确率**

# 将预测值和真实值转换为二分类标签（大于0为1， 否则为0）
predicted_labels = (predicted_values > 0).astype(int)
true_labels = (true_values > 0).astype(int)

# 检查标签的形状
print("二分类预测标签形状:", predicted_labels.shape)
print("二分类真实标签形状:", true_labels.shape)

# 计算精确率
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
print(f"预测结果大于0时的精确率：{precision:.4f}")

# 打印分类报告
print("分类报告：\n", classification_report(true_labels, predicted_labels))

# 绘制训练和验证损失
plt.plot(loss_logger.train_loss, label='训练损失')
plt.plot(loss_logger.val_loss, label='验证损失')
plt.legend()
plt.show()

# **修改后的绘图部分：处理多变量 TimeSeries**

# 获取 TimeSeries 的维度
num_components = data["test"].n_components

# 拆分多变量 TimeSeries 为单变量列表
true_univariates = [data["test"][-pred_steps:].univariate_component(i) for i in range(num_components)]
pred_univariates = [pred_series.univariate_component(i) for i in range(num_components)]

# 获取组件名称（如果有）
component_names = data["test"].components

# 确保组件名称可用
if component_names.empty:
    component_names = [f"Component {i + 1}" for i in range(num_components)]
else:
    # 将组件名称转换为列表以便后续使用
    component_names = component_names.tolist()

# 绘制前3个组件的实际数据和预测结果
for i in range(min(3, num_components)):
    true_series = true_univariates[i]
    pred_series_univar = pred_univariates[i]
    series_name = component_names[i] if i < len(component_names) else f"Component {i + 1}"

    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(
        true_series.pd_series(),  # 单变量 TimeSeries 的 pd_series()
        label=f"{series_name} 实际数据"
    )
    plt.plot(
        pred_series_univar.pd_series(),
        label=f"{series_name} 预测结果"
    )
    plt.legend()
    plt.title(f"{series_name} 实际 vs 预测")
    plt.xlabel("时间")
    plt.ylabel("值")
    plt.show()


# 定义用于超参数优化的目标函数
def objective(trial):
    input_chunk_length = trial.suggest_int("input_chunk_length", 1, 128)
    output_chunk_length = trial.suggest_int("output_chunk_length", 1, 20)

    # 使用建议的超参数初始化模型
    model = NLinearModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        pl_trainer_kwargs=get_pl_trainer_kwargs(),
        work_dir=work_dir,
        save_checkpoints=True,
        force_reset=True,
        model_name=model_name,
        # loss_fn=MSELoss()  # 使用均方误差损失函数（可选）
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

    # 预测步骤
    pred_steps = TIMESERIES_LENGTH["test_length"]
    pred_input = data["test"][:-pred_steps]

    # 进行预测
    pred_series = model.predict(n=pred_steps, series=pred_input)

    # 获取预测值和真实值
    predicted = pred_series.values()
    true = data["test"][-pred_steps:].values()

    # 计算均方误差作为优化目标（越小越好）
    mse = mean_squared_error(true, predicted)
    print(f"均方误差 (MSE)：{mse:.4f}")

    return mse  # 返回均方误差作为优化目标


def delete_study(study_name, storage_url):
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"成功删除 study：{study_name}")
    except KeyError:
        print(f"找不到 study：{study_name}")


if __name__ == '__main__':
    delete_study("nlinearmodel-optimization", "sqlite:///data/optuna/optuna_study.db")

    # 创建一个新的 Optuna study 并开始优化
    study = optuna.create_study(
        direction="minimize",  # 对于 MSE，我们希望最小化
        study_name="nlinearmodel-optimization",
        storage="sqlite:///data/optuna/optuna_study.db",  # 指定 SQLite 数据库位置
        load_if_exists=True  # 如果 study 已存在，则加载它
    )
    study.optimize(objective, n_trials=50)  # 调整试验次数

    print("最佳超参数: ", study.best_params)
    print("最佳均方误差: ", study.best_value)