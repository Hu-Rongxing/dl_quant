import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from typing import Tuple
import matplotlib


# 导入您的自定义函数
# 假设这些函数在同一文件或已正确导入
from load_data.multiple_timeseries import load_and_clean_data, detect_turning_points
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

def detect_turning_points(prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用局部极值检测转折点。

    Args:
        prices (np.ndarray): 价格序列。
        order (int): 检测的窗口大小。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 局部最大值和最小值的索引。
    """
    local_max = argrelextrema(prices, np.greater, order=order)[0]
    local_min = argrelextrema(prices, np.less, order=order)[0]
    return local_max, local_min


def plot_turning_points(dataframe: pd.DataFrame, stock_code: str, order: int = 5):
    """
    绘制股票收盘价曲线，并标记局部最大值和最小值。

    Args:
        dataframe (pd.DataFrame): 包含股票数据的 DataFrame。
        stock_code (str): 要绘制的股票代码。
        order (int): 检测转折点的窗口大小。
    """
    # 选择特定股票的数据
    stock_data = dataframe[dataframe['stock_code'] == stock_code].reset_index(drop=True)

    # 获取价格序列
    prices = stock_data['close'].values

    # 检测转折点
    local_max, local_min = detect_turning_points(prices, order=order)

    # 创建绘图
    plt.figure(figsize=(14, 7))

    # 绘制收盘价曲线
    plt.plot(stock_data.index, prices, label='收盘价', color='blue')

    # 标记局部最大值
    plt.scatter(stock_data.index[local_max], prices[local_max], color='red', marker='^', label='局部最大值')

    # 标记局部最小值
    plt.scatter(stock_data.index[local_min], prices[local_min], color='green', marker='v', label='局部最小值')

    # 添加标题和标签
    plt.title(f'股票代码 {stock_code} 的收盘价及转折点')
    plt.xlabel('时间索引')
    plt.ylabel('收盘价')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 展示图形
    plt.show()


if __name__ == '__main__':
    # 加载并清理数据
    clean_data = load_and_clean_data()

    # 列出所有可用的股票代码
    available_stocks = clean_data['stock_code'].unique()
    print("可用的股票代码：", available_stocks)

    # 选择要绘制的股票代码（从可用的股票代码中选择一个）
    for stock_code in available_stocks:

        # 调用绘图函数，order 参数可根据需要调整
        plot_turning_points(clean_data, stock_code, order=5)