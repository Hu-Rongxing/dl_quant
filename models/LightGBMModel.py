#!/usr/bin/env python
# coding: utf-8

import torch
from darts.models import LightGBMModel  # 导入 LightGBM 模型
from sklearn.metrics import precision_score  # 用于计算精确率
import matplotlib.pyplot as plt  # 用于绘图
import optuna  # 用于超参数优化
from pathlib import Path  # 用于处理文件路径
import numpy as np
import matplotlib

# 自定义设置
from config import TIMESERIES_LENGTH  # 导入时间序列长度配置
from load_data.multivariate_timeseries import generate_processed_series_data  # 导入数据加载函数
from utils.logger import logger  # 导入日志记录器

# 设置浮点数矩阵乘法精度
torch.set_float32_matmul_precision('medium')

# 常量定义
MODEL_NAME = "LightBGMModel"  # 模型名称
WORK_DIR = Path(f"logs/{MODEL_NAME}_logs").resolve()  # 工作目录
PRED_STEPS = TIMESERIES_LENGTH["test_length"]  # 预测步长
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 准备训练和验证数据
data = generate_processed_series_data('training')

# 定义设备 (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义模型
def define_model(trial):
    future_lags = trial.suggest_int('lags_future_covariates', 1, 21)

    base_parameters = {
        'lags': trial.suggest_int("lags", 1, 48),
        'lags_past_covariates': trial.suggest_int('lags_past_covariates', 1, 48),
        'lags_future_covariates': [-i for i in range(future_lags)],
        'output_chunk_length': 1,
        'output_chunk_shift': 0,
        'add_encoders': None,
        'likelihood': None,
        'quantiles': None,
        'multi_models': trial.suggest_categorical('multi_models', [True, False])
    }

    lgb_parameters = {
        'num_leaves': trial.suggest_int("num_leaves", 15, 31),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int("n_estimators", 100, 300),
        'min_child_samples': trial.suggest_int("min_child_samples", 2, 20),
        'min_child_weight': trial.suggest_float("min_child_weight", 0.001, 0.1),
        'min_gain_to_split': 0,
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int("max_depth", 3, 8),
        'max_bin': trial.suggest_int("max_bin", 200, 300),
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'verbose': -1
    }

    parameters = {**base_parameters, **lgb_parameters}
    return LightGBMModel(**parameters)


def train_and_evaluate(model, data):
    """训练和评估 LightGBM 模型"""
    precision = 0.0
    try:
        model.fit(
            series=data['train'][-300:],
            past_covariates=data['past_covariates'],
            future_covariates=data['future_covariates'],
            val_series=data['val'],
            val_past_covariates=data['past_covariates'],
            val_future_covariates=data['future_covariates'],
        )

        # 使用 backtest 进行回测
        backtest_series = model.historical_forecasts(
            series=data['test'],
            past_covariates=data['past_covariates'],
            future_covariates=data['future_covariates'],
            start=data['test'].time_index[-PRED_STEPS],
            forecast_horizon=1,
            stride=1,
            retrain=False
        )

        true_labels = data["test"][-PRED_STEPS:].values().flatten().astype(int)
        probabilities = backtest_series[-PRED_STEPS:].values().flatten()
        binary_predictions = (probabilities > 0.5).astype(int)

        precision = precision_score(true_labels, binary_predictions)

        # 绘图
        plt.figure(figsize=(12, 6))
        data["test"][-PRED_STEPS * 2:].plot(label='实际值')
        backtest_series[-PRED_STEPS * 2:].plot(label='预测值', lw=2, color="red", alpha=0.6)
        plt.title(f"LightGBM 预测结果 (精确率: {precision:.4f})")
        plt.legend()
        plt.show()
        plt.close()

    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")

    # 清理内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return precision


def check_data_quality(data):
    """检查数据质量"""
    logger.info("检查数据质量...")
    for key in ['train', 'val', 'test']:
        if key in data:
            series = data[key]
            values = series.values()
            logger.info(f"\n{key} 数据统计:")
            logger.info(f"形状: {values.shape}")
            logger.info(f"唯一值数量: {len(np.unique(values))}")
            logger.info(f"最小值: {np.min(values)}")
            logger.info(f"最大值: {np.max(values)}")
            logger.info(f"均值: {np.mean(values)}")
            logger.info(f"标准差: {np.std(values)}")
            logger.info(f"缺失值数量: {np.isnan(values).sum()}")
            if len(values.shape) > 1:
                constant_cols = np.where(np.std(values, axis=0) == 0)[0]
                if len(constant_cols) > 0:
                    logger.warning(f"发现常量列: {constant_cols}")

    if 'past_covariates' in data:
        logger.info("\n过去协变量统计:")
        logger.info(f"形状: {data['past_covariates'].values().shape}")

    if 'future_covariates' in data:
        logger.info("\n未来协变量统计:")
        logger.info(f"形状: {data['future_covariates'].values().shape}")

def objective(trial):
    model = define_model(trial)  # 创建模型
    precision = train_and_evaluate(model, data)  # 训练并评估模型
    logger.info(f"试验{trial.number}: 最佳准确率: {study.best_value:.4%}")  # 记录最佳精确率
    logger.info(f"当前准确率:{precision:.4%}；当前超参数： {trial.params}")  # 记录当前超参数
    return precision  # 返回精确率作为优化目标

if __name__ == '__main__':
    # 检查数据质量
    check_data_quality(data)

    # 创建工作目录
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 设置优化器
    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm-precision-optimization",
        storage="sqlite:///data/optuna/optuna_study.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 添加提前停止
    study.optimize(
        objective,  # 修正为 objective
        n_trials=50,
        n_jobs=1,
        # callbacks=[optuna.callbacks.ProgressBar()],
        catch=(Exception,)
    )

    # 输出结果
    logger.info("优化完成!")
    logger.info(f"最佳超参数: {study.best_params}")
    logger.info(f"最佳精确率: {study.best_value:.4f}")

    # 可视化优化结果
    try:
        optuna.visualization.plot_optimization_history(study)
        plt.show()
        optuna.visualization.plot_param_importances(study)
        plt.show()
    except Exception as e:
        logger.error(f"可视化过程中出现错误: {str(e)}")

    # 最佳超参数: {'lags_future_covariates': 5, 'lags': 11, 'lags_past_covariates': 28, 'multi_models': False, 'num_leaves': 27, 'learning_rate': 0.01659083896649441, 'n_estimators': 113, 'min_child_samples': 7, 'min_child_weight': 0.039948756447233384, 'subsample': 0.9074547873450443, 'colsample_bytree': 0.6969909500680137, 'reg_alpha': 1.015005223943824e-08, 'reg_lambda': 3.046135392550891e-05, 'max_depth': 6, 'max_bin': 201}
