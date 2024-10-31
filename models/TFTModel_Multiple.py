# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from darts import TimeSeries
from darts.models import TFTModel
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # 导入新的评价指标
from sklearn.utils.class_weight import compute_class_weight  # 用于计算类别权重
from torch.nn import CrossEntropyLoss
from load_data.multiple_timeseries import prepare_timeseries_data

from models.params import get_pl_trainer_kwargs

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

def main():
    # 加载数据
    result = prepare_timeseries_data()
    stock_code = list(result.keys())[2]

    train_series = result[stock_code]['train']
    val_series = result[stock_code]['val']
    test_series = result[stock_code]['test']

    past_covariates_train = result[stock_code]['past_covariates']
    past_covariates_val = result[stock_code]['past_covariates']
    future_covariates_train = result[stock_code]['future_covariates']
    future_covariates_val = result[stock_code]['future_covariates']

    # 提取训练集的真实标签，用于计算类别权重
    train_values = train_series.values()
    train_labels = np.argmax(train_values, axis=1)

    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = class_weights.astype(np.float32)
    print(f'类别权重：{class_weights}')

    # 将类别权重转换为张量，在损失函数中使用
    import torch
    weight_tensor = torch.tensor(class_weights)

    # 定义加权的交叉熵损失函数
    loss_fn = CrossEntropyLoss(weight=weight_tensor)

    # 定义模型，使用加权的损失函数
    model = TFTModel(
        input_chunk_length=30,
        output_chunk_length=7,
        hidden_size=96,  # 可以适当增大隐藏层大小
        lstm_layers=3,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=64,  # 增大 batch_size
        n_epochs=100,
        random_state=42,
        pl_trainer_kwargs=get_pl_trainer_kwargs(full_training=True),
        # loss_fn=loss_fn
    )

    # 训练模型
    model.fit(
        series=train_series,
        past_covariates=past_covariates_train,
        future_covariates=future_covariates_train,
        val_series=val_series,
        val_past_covariates=past_covariates_val,
        val_future_covariates=future_covariates_val,
        verbose=True
    )

    # 使用测试集进行预测
    combined_series = train_series.concatenate(val_series)


    forecast = model.predict(
        n=len(test_series),
        series=combined_series,
        past_covariates=past_covariates_val,
        future_covariates=future_covariates_val,
        num_samples=1
    )

    # 处理预测结果
    forecast_values = forecast.all_values(copy=False)
    # 注意，由于模型输出的是 logits，所以直接使用 softmax
    probabilities = softmax(forecast_values, axis=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    # 提取真实标签
    true_values = test_series.all_values(copy=False)
    true_labels = np.argmax(true_values, axis=1)

    # 计算评价指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
    recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
    f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
    print(f'预测准确率：{accuracy * 100:.2f}%')
    print(f'预测精确率：{precision * 100:.2f}%')
    print(f'召回率：{recall * 100:.2f}%')
    print(f'F1分数：{f1 * 100:.2f}%')

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(test_series.time_index, true_labels, label='真实标签')
    plt.plot(forecast.time_index, predicted_labels, label='预测标签', linestyle='--')
    plt.legend()
    plt.title(f'股票代码 {stock_code} 的分类预测结果')
    plt.xlabel('时间')
    plt.ylabel('类别标签')
    plt.show()

if __name__ == '__main__':
    main()