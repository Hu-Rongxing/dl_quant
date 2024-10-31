
#!/usr/bin/env python
# coding: utf-8

import torch
from darts.models import LightGBMModel  # 导入 LightGBMModel
from sklearn.metrics import precision_score  # 用于计算精确率
import matplotlib.pyplot as plt  # 用于绘图
import optuna  # 用于超参数优化
from pathlib import Path  # 用于处理文件路径
import numpy as np  # 用于数值计算

# 自定义设置
from config import TIMESERIES_LENGTH  # 导入时间序列长度配置
from load_data.multivariate_timeseries import generate_processed_series_data  # 导入数据加载函数
from utils.logger import logger  # 导入日志记录器

# 常量定义
MODEL_NAME = "LightGBMModel"  # 模型名称
WORK_DIR = Path(f"logs/{MODEL_NAME}_logs").resolve()  # 工作目录
PRED_STEPS = TIMESERIES_LENGTH["test_length"]  # 预测步长

# 准备训练和验证数据 (在循环外加载数据)
data = generate_processed_series_data('training')  # 假设返回的数据包含train/test/past_covariates/future_covariates

# 定义模型
def define_model(trial):
    """
    定义 LightGBMModel 并根据 Optuna Trial 建议的参数进行初始化。

    Args:
        trial: Optuna Trial 对象，用于建议超参数。

    Returns:
        LightGBMModel: 初始化的 LightGBMModel。
    """
    # 模型参数
    lags = trial.suggest_int("lags", 1, 64)
    lags_past_covariates = trial.suggest_int("lags_past_covariates", 1, 64)
    lags_future_covariates = [0, -1, -2]
    output_chunk_length = trial.suggest_int("output_chunk_length", 1, min(20, PRED_STEPS))

    # LightGBM 回归器的参数
    lgbm_params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "binary",
        "random_state": 42,
    }

    model = LightGBMModel(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
        verbose=-1,  # 关闭 LightGBM 的输出
        **lgbm_params
    )

    return model

def train_and_evaluate(model, data):
    """
    训练和评估 LightGBMModel。

    Args:
        model: LightGBMModel。
        data: 包含训练、验证和测试数据的数据字典。

    Returns:
        float: 精确率。
    """
    # 训练模型
    model.fit(
        series=data['train'][-300:],  # 使用部分训练数据
        past_covariates=data['past_covariates'],  # 过去的协变量
        future_covariates=data['future_covariates'],  # 未来的协变量
        val_series=data['val'],  # 验证数据
        val_past_covariates=data['past_covariates'],  # 验证集过去的协变量
        val_future_covariates=data['future_covariates'],  # 验证集未来的协变量
    )

    # 生成预测
    forecast = model.predict(
        n=PRED_STEPS,
        series=data['train'],
        past_covariates=data.get('past_covariates_test', None),
        future_covariates=data.get('future_covariates_test', None),
    )

    # 计算精确度
    true_labels = data["test"][-PRED_STEPS:] # 真实标签
    print(true_labels.time_index)
    true_labels = true_labels.values().flatten().astype(int)
    probabilities = forecast.values().flatten()  # 预测概率
    binary_predictions = (probabilities > 0.5).astype(int)  # 二元预测
    print(forecast.time_index)

    precision = precision_score(true_labels, binary_predictions)  # 计算精确率
    logger.info(f"精度: {precision:.2%}")

    # 绘图 (可根据需要取消注释)
    data["test"][-PRED_STEPS:].plot(label='实际值')
    forecast.plot(label='预测值', lw=3, color="red", alpha=0.5)
    plt.title("LightGBMModel 预测结果")
    plt.legend()
    plt.show()
    plt.close()

    return precision

def objective(trial):
    """
    Optuna 的目标函数，用于优化超参数。

    Args:
        trial: Optuna Trial 对象。

    Returns:
        float: 精确率，作为优化的目标。
    """
    model = define_model(trial)  # 定义模型
    precision = train_and_evaluate(model, data)  # 训练和评估模型
    logger.info(f"试验 {trial.number}: 精确率: {precision}")
    logger.info(f"当前超参数： {trial.params}")
    return precision

if __name__ == '__main__':
    study = optuna.create_study(
        direction="maximize",  # 最大化精确率
        study_name="lightgbmmodel-precision-optimization",  # 研究名称
        storage="sqlite:///data/optuna/optuna_study.db",  # 数据库路径
        load_if_exists=True  # 如果数据库存在则加载
    )
    study.optimize(objective, n_trials=50, n_jobs=1)  # 开始优化

    logger.info(f"最佳超参数: {study.best_params}")  # 输出最佳超参数
    logger.info(f"最佳精确率: {study.best_value:.4f}")  # 输出最佳精确率

