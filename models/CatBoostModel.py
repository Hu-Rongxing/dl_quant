#!/usr/bin/env python
# coding: utf-8

from darts.models import XGBModel  # 导入 XGBModel
from darts.utils.callbacks import TQDMProgressBar  # 使用正确的进度条类名
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import optuna
from config import TIMESERIES_LENGTH
from load_data.multivariate_timeseries import prepare_timeseries_data
from utils.model import LossLogger

# 初始化损失记录器和进度条
loss_logger = LossLogger()
progress_bar = TQDMProgressBar()  # 修正类名

# 准备训练和验证数据
data = prepare_timeseries_data('training')

# 初始化 XGBModel，添加必要的参数
model = XGBModel(
    lags=3,  # 设置适当的滞后期
    lags_past_covariates=[-1],  # 设置过去协变量的滞后期
    lags_future_covariates=[0],  # 设置未来协变量的滞后期
    # objective='multi:softprob',  # 设置 XGBoost 为多分类目标
    num_class=3,  # 类别数量
    random_state=42,  # 设置随机种子
)

# 训练模型
model.fit(
    series=data['train'],
    past_covariates=data['past_covariates'],
    future_covariates=data['future_covariates'],
    val_series=data['val'],
    val_past_covariates=data['past_covariates'],
    val_future_covariates=data['future_covariates'],
    verbose=True,
    # callbacks=[progress_bar],  # 添加进度条回调
)

# 进行预测
pred_steps = TIMESERIES_LENGTH["test_length"]
pred_input = data["test"][:-pred_steps]

# 进行预测
pred_series = model.predict(n=pred_steps, series=pred_input)

# 获取预测值和真实值
predicted_probs = pred_series.values()
true_values = data["test"][-pred_steps:].values()

# 将预测概率转换为预测标签
predicted_labels = np.argmax(predicted_probs, axis=1)
true_labels = true_values.astype(int).flatten()

# 输出分类报告和准确率
print("分类报告：\n", classification_report(true_labels, predicted_labels))
print("准确率:", accuracy_score(true_labels, predicted_labels))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = [-1, 0, 1]
plt.xticks([0, 1, 2], tick_marks)
plt.yticks([0, 1, 2], tick_marks)
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 定义用于超参数优化的目标函数
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    subsample = trial.suggest_uniform('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)

    # 使用建议的超参数初始化模型
    model = XGBModel(
        lags=3,
        lags_past_covariates=[-1],
        lags_future_covariates=[0],
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='multi:softprob',
        num_classes=3,
        random_state=42,
    )

    # 训练模型
    model.fit(
        series=data['train'],
        past_covariates=data['past_covariates'],
        future_covariates=data['future_covariates'],
        val_series=data['val'],
        val_past_covariates=data['past_covariates'],
        val_future_covariates=data['future_covariates'],
        verbose=False,
        callbacks=[progress_bar],  # 添加进度条回调
    )

    # 进行预测
    pred_steps = TIMESERIES_LENGTH["test_length"]
    pred_input = data["test"][:-pred_steps]

    pred_series = model.predict(n=pred_steps, series=pred_input)

    predicted_probs = pred_series.values()
    true_values = data["test"][-pred_steps:].values()

    predicted_indices = np.argmax(predicted_probs, axis=1)
    true_indices = true_values.astype(int).flatten()

    predicted_labels = [inverse_label_mapping[idx] for idx in predicted_indices]
    true_labels = [inverse_label_mapping[idx] for idx in true_indices]

    # 计算准确率作为优化目标（越大越好）
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"本次试验的准确率: {accuracy:.4f}")

    return accuracy  # 返回准确率

def delete_study(study_name, storage_url):
    try:
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"成功删除 study：{study_name}")
    except KeyError:
        print(f"找不到 study：{study_name}")

if __name__ == '__main__':
    delete_study("xgbmodel-optimization", "sqlite:///data/optuna/optuna_study.db")

    # 创建一个新的 Optuna study 并开始优化
    study = optuna.create_study(
        direction="maximize",
        study_name="xgbmodel-optimization",
        storage="sqlite:///data/optuna/optuna_study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)

    print("最佳超参数: ", study.best_params)
    print("最佳准确率: ", study.best_value)
