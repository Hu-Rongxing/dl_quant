#!/usr/bin/env python
# coding: utf-8

import torch
from darts.models import TFTModel  # 导入 TFT 模型
from pathlib import Path  # 用于处理文件路径
import matplotlib.pyplot as plt
import datetime

# 自定义设置
from config import TIMESERIES_LENGTH  # 导入时间序列长度配置
from load_data.multivariate_timeseries import generate_processed_series_data  # 导入数据加载函数
from utils.logger import logger  # 导入日志记录器
from models.params import get_pl_trainer_kwargs, loss_logger  # 导入训练参数配置函数

# 设置浮点数矩阵乘法精度，以提高计算性能
torch.set_float32_matmul_precision('medium')

# 常量定义
MODEL_NAME = "TFTModel"  # 模型名称
WORK_DIR = Path(f"logs/{MODEL_NAME}_logs").resolve()  # 工作目录
PRED_STEPS = TIMESERIES_LENGTH["test_length"]  # 预测步长
MODEL_PATH = WORK_DIR / "best_tft_model.pth"


def fit_model():
    # 准备预测数据 (加载用于预测的数据集)
    data = generate_processed_series_data('predicting')

    # 定义设备 (如果有 GPU 可用则使用，否则使用 CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数 (根据之前的超参数优化结果)
    parameters = {
        'input_chunk_length': 64,  # 输入序列长度
        'output_chunk_length': 9,  # 输出序列长度
        'hidden_size': 19,  # 隐藏层大小
        'dropout': 0.1,  # dropout 率
        'lstm_layers': 1,  # LSTM 层数
        'num_attention_heads': 2,  # 注意力头的数量
        'full_attention': False,  # 是否使用全注意力机制
        'feed_forward': 'GatedResidualNetwork',  # 前馈网络类型
        'hidden_continuous_size': 24  # 连续变量隐藏层大小
    }

    # 初始化 TFT 模型
    model = TFTModel(
        **parameters,
        pl_trainer_kwargs=get_pl_trainer_kwargs(full_training=True),  # 获取 PyTorch Lightning 训练参数
        work_dir=WORK_DIR,  # 设置工作目录
        save_checkpoints=True,  # 训练过程中保存检查点
        force_reset=True,  # 强制重置模型（如果已有同名模型将被覆盖）
        model_name=MODEL_NAME,  # 模型名称
        batch_size=128,  # 批量大小
        n_epochs=50,  # 训练轮数
        random_state=42,  # 随机种子，保证结果可复现
        log_tensorboard=False,  # 是否记录到 TensorBoard
    )

    # 训练模型
    model.fit(
        series=data['train'],  # 训练数据序列
        past_covariates=data['past_covariates'],  # 过去的协变量
        future_covariates=data['future_covariates'],  # 未来的协变量
        val_series=data['val'],  # 验证数据序列
        val_past_covariates=data['past_covariates'],  # 验证集过去的协变量
        val_future_covariates=data['future_covariates'],  # 验证集未来的协变量
    )


    # 保存最佳模型
    model.save(str(MODEL_PATH))  # 保存模型到指定路径

    # 损失可视化
    # 训练结束后绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(loss_logger.train_loss, label='Training Loss')
    plt.plot(loss_logger.val_loss, label='Validation Loss')
    plt.title(f'Training and Validation Loss 【{datetime.datetime.today()}】')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()

    return model


def predict_market():
    logger.info("下载预测数据")
    data = generate_processed_series_data('predicting')
    # 读取最佳模型
    logger.info("读取最佳模型。")
    model = TFTModel.load(str(MODEL_PATH))  # 从指定路径加载模型

    # 进行预测
    logger.info("预测……")
    forecast = model.predict(
        n=1,  # 预测的步数
        series=data['train'],  # 用于生成预测的输入序列
        past_covariates=data['past_covariates'],  # 过去的协变量
        future_covariates=data['future_covariates'],  # 未来的协变量
    )

    # 处理预测结果
    logger.info("生成买入列表")
    result = forecast.pd_dataframe().T  # 将预测结果转换为 DataFrame 并转置
    result = result.iloc[:, 0].sort_values(ascending=False)  # 对第一列数据按降序排序
    high_confidence_indices = result[result > 0.5].index.to_list()  # 提取预测值大于 0.5 的索引列表

    # 输出高置信度的预测结果索引
    logger.info(f"预测值大于 0.5 的索引列表: {high_confidence_indices}")
    return high_confidence_indices


if __name__ == '__main__':
    # 主程序入口，如果需要，可以在这里添加其他执行代码
    fit_model()
    result = predict_market()
    print(result)
