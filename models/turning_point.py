import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 请根据您系统中的字体进行调整，例如 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv('data/stock_data.csv', encoding='utf-8', dtype={"time": str})

# 将时间列转换为 datetime 类型
if df['time'].str.isnumeric().all():
    df['time'] = pd.to_datetime(df['time'].astype(float) / 1000, unit='s')
else:
    df['time'] = pd.to_datetime(df['time'])

# 重置索引为整数索引
df.reset_index(drop=True, inplace=True)


def detect_turning_points(prices, order=5):
    # 使用局部极值检测
    local_max = argrelextrema(prices, np.greater, order=order)[0]
    local_min = argrelextrema(prices, np.less, order=order)[0]

    return local_max, local_min


def identify_reversal_points(df):
    stocks = df['stock_code'].unique()

    # 初始化一个 target 列，全设为 0
    df['target'] = 0

    for stock in stocks:
        stock_data = df[df['stock_code'] == stock]
        stock_data = stock_data.sort_values('time')
        stock_data = stock_data.dropna(subset=['close'])
        prices = stock_data['close'].values

        if len(prices) < 3:
            continue

            # 检测转折点
        local_max, local_min = detect_turning_points(prices)

        # 筛选最近的 220 天数据
        recent_days = stock_data[stock_data['time'] >= (stock_data['time'].max() - pd.Timedelta(days=220))]

        # 重新获取最近价格
        recent_prices = recent_days['close'].values

        # 检查最近数据是否足够
        if len(recent_prices) < 3:
            print(f'警告：股票 {stock} 的最近 220 天数据不足。')
            continue

            # 将转折点标记到 df 中
        for idx in local_max:
            df.loc[(df['stock_code'] == stock) & (df['time'] == stock_data.iloc[idx]['time']), 'target'] = 1
        for idx in local_min:
            df.loc[(df['stock_code'] == stock) & (df['time'] == stock_data.iloc[idx]['time']), 'target'] = -1

            # 绘制价格和转折点
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(recent_prices)), recent_prices, label='价格')

        # 绘制转折点
        plt.plot(local_max[local_max >= (len(prices) - len(recent_prices))] - (len(prices) - len(recent_prices)),
                 recent_prices[local_max[local_max >= (len(prices) - len(recent_prices))] - (
                         len(prices) - len(recent_prices))],
                 'gv', label='多头转折点')
        plt.plot(local_min[local_min >= (len(prices) - len(recent_prices))] - (len(prices) - len(recent_prices)),
                 recent_prices[local_min[local_min >= (len(prices) - len(recent_prices))] - (
                         len(prices) - len(recent_prices))],
                 'r^', label='空头转折点')

        plt.title(f'{stock} 价格和转折点（最近 220 天）')
        plt.xlabel('时间索引')
        plt.ylabel('价格')
        plt.legend()
        plt.xlim(0, len(recent_prices) - 1)  # 设置 x 轴范围
        plt.xticks(rotation=45)  # 旋转 x 轴标签以便于阅读
        plt.tight_layout()  # 自动调整布局以避免重叠
        plt.savefig(f'data/picture/{stock}_price_reversals_last_220_days.png')
        plt.close()

        # 返回更新后的 DataFrame
    return df


if __name__ == '__main__':
    # 调用函数并获取更新后的 DataFrame
    updated_df = identify_reversal_points(df)
    # 保存更新后的 DataFrame
    updated_df.to_csv('data/updated_stock_data.csv', index=False, encoding='utf-8')