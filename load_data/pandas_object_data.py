import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from utils.logger import logger
from .download_xt_data import get_data_from_local


def calculate_technical_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标，并处理缺失值"""
    group = group.bfill().ffill()  # 前向填充后向填充

    # 计算未来三天的最高价和最低价
    future_high = group['high'].rolling(window=3, min_periods=1).max().shift(-2)
    future_low = group['low'].rolling(window=3, min_periods=1).min().shift(-2)

    # 计算未来涨幅和跌幅
    max_increase = (future_high - group['close']) / group['close']
    min_decrease = (group['close'] - future_low) / group['close']

    # 生成目标变量
    group['target'] = ((max_increase > 0.01) & (min_decrease > -0.01)).astype(int)

    # 计算移动平均线和指数移动平均线
    ma_windows = [3, 5, 10]
    for window in ma_windows:
        group[f'ma_{window}'] = group['close'].rolling(window=window).mean()

    group['ema_5'] = group['close'].ewm(span=12, adjust=False).mean()
    group['ema_10'] = group['close'].ewm(span=26, adjust=False).mean()

    # 计算平均价格
    group['ave_price'] = np.where(group['volume'] == 0, 0, group['amount'] / (group['volume'] * 100))

    return group.bfill().ffill()  # 最后再进行一次填充


def process_dataframe() -> Tuple[pd.DataFrame, Dict[Any, int]]:
    """加载并处理原始数据，计算技术指标"""
    try:
        dataframe = get_data_from_local()
        dataframe.sort_values(by=['stock_code', 'time'], ascending=True, inplace=True)
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 生成整数序列
        dataframe['time_seq'] = dataframe['time'].factorize()[0]

        # 计算技术指标
        dataframe = (dataframe.groupby('stock_code')
                     .apply(calculate_technical_indicators, include_groups=False)
                     .reset_index(drop=False))

        return dataframe, {time: seq for seq, time in enumerate(dataframe['time'].unique())}

    except Exception as e:
        logger.error(f"数据处理失败 - {str(e)}")
        raise


def generate_wide_dataframe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成宽格式数据框并包含协变量"""
    data_cleaned, date_to_int_mapper = process_dataframe()
    # data_cleaned.reset_index(level='date', inplace=True)

    data_cleaned.to_csv("data/precessed_data/data.csv", index=False)  # 保存数据

    # 准备特征变量的列名
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'ma_3', 'ma_5', 'ma_10', 'ave_price']

    # 创建目标变量和协变量数据框
    target_df = data_cleaned.pivot(index="time_seq", columns="stock_code", values='target')

    covariate_df = data_cleaned.pivot_table(index="time_seq", columns="stock_code", values=feature_columns)
    covariate_df.columns.names = ['variable', 'stock_code']

    # 添加指数数据作为协变量
    stock_list = ['000001.SH', '399001.SZ', '399006.SZ']
    index_data = get_data_from_local(stock_list=stock_list).reset_index()
    index_data['time_seq'] = index_data['time'].map(date_to_int_mapper)

    index_data_pivot = index_data.pivot(index="time_seq", columns="stock_code",
                                        values=['open', 'high', 'low', 'close', 'amount'])
    index_data_pivot.columns.names = ['variable', 'stock_code']

    covariate_df = covariate_df.join(index_data_pivot)  # 合并数据框
    covariate_df.columns = ["_".join([n,c]).replace('.','_') for n,c in covariate_df.columns]

    return target_df, covariate_df