import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from xtquant import xtdata
from pickle import dump, load
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Dict

from config import DATA_SAVE_PATHS, TIMESERIES_LENGTH
from utils.logger import logger
from load_data.download_xt_data import get_data_from_local


def calculate_technical_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标，并处理缺失值"""
    group = group.bfill().ffill()  # 先前向填充，再后向填充

    # 计算未来三天的最高价和最低价
    future_high = group['high'].rolling(window=3, min_periods=1).max().shift(-2)
    future_low = group['low'].rolling(window=3, min_periods=1).min().shift(-2)

    # 计算未来三天的最大涨幅和最小跌幅
    max_increase = (future_high - group['close']) / group['close']
    min_decrease = (group['close'] - future_low) / group['close']

    # 生成目标变量
    group['target'] = ((max_increase > 0.01) & (min_decrease > -0.01)).astype(int)

    # 计算移动平均线 (MA)
    group['ma_3'] = group['close'].rolling(window=7).mean()
    group['ma_5'] = group['close'].rolling(window=14).mean()
    group['ma_10'] = group['close'].rolling(window=21).mean()

    # 计算指数移动平均线 (EMA)
    group['ema_5'] = group['close'].ewm(span=12, adjust=False).mean()
    group['ema_10'] = group['close'].ewm(span=26, adjust=False).mean()

    # 平均价格
    # 计算平均价格 ave_price
    group['ave_price'] = np.where(group['volume'] == 0, 0, group['amount'] / (group['volume'] * 100))
    group = group.bfill().ffill()

    return group


def process_dataframe() -> pd.DataFrame:
    """加载并处理原始数据，计算技术指标"""
    try:
        dataframe = get_data_from_local()
        dataframe.sort_values(by=['stock_code', 'time'], ascending=True, inplace=True)
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 生成整数序列
        date_to_int_mapper = {d: i for i, d in enumerate(dataframe['time'].unique())}
        dataframe['time_seq'] = dataframe['time'].map(date_to_int_mapper)

        # 按股票代码分组，计算技术指标并填充缺失值
        dataframe = (dataframe.groupby('stock_code')
                     .apply(calculate_technical_indicators, include_groups=False)
                     .reset_index("stock_code", drop=False))

        return dataframe

    except Exception as e:
        logger.error(f"数据处理失败 - {str(e)}")
        raise


def generate_processed_series_data(mode: str = 'training') -> Dict[str, TimeSeries]:
    """
    生成处理后的时间序列数据，包括训练集、验证集、测试集和协变量。

    Args:
        mode (str, optional): 数据模式，'training' 或 'predicting'。 Defaults to 'training'.

    Returns:
        Dict[str, TimeSeries]: 包含处理后时间序列数据的字典。
    """
    try:
        if mode not in ['training', 'predicting']:
            raise ValueError("模式必须是 'training' 或 'predicting'。")

        data_cleaned = process_dataframe()
        data_cleaned.reset_index(level='date', inplace=True)

        data_cleaned.to_csv("data/precessed_data/data.csv")  # 保存清洗后的数据到 CSV 文件

        # 准备 Darts TimeSeries 的数据
        feature_columns = ['open', 'high', 'low', 'close', 'volume',
       'amount', 'ma_3', 'ma_5', 'ma_10', 'ave_price']  # 模型使用的特征

        # 创建目标变量和协变量数据框
        target_df = data_cleaned.pivot(
            index="time_seq",
            columns="stock_code",
            values='target') # 目标变量数据框

        # TODO: 在这里增加指数数据

        covariate_df = data_cleaned.pivot_table(
            index="time_seq",
            columns="stock_code",
            values=feature_columns
        )  # 协变量数据框
        covariate_df.columns.names = ['variable', 'stock_code']  # 设置列名

        # 转换为 Darts TimeSeries 对象
        target_time_series = TimeSeries.from_dataframe(
            df = target_df,
            time_col=None,
            value_cols=None,
            fill_missing_dates=True,
            freq=1,
            fillna_value=0
        ).astype(np.float32)  # 目标时间序列

        past_covariate_time_series = TimeSeries.from_dataframe(
            df = covariate_df,
            time_col=None,
            value_cols=None,
            fill_missing_dates=True,
            freq=1,
            fillna_value=0
        ).astype(np.float32)  # 协变量时间序列

        # 将数据分割成训练集、验证集和测试集
        time_params = TIMESERIES_LENGTH  # 加载时间序列参数
        train_index = slice(time_params['header_length'], -(time_params['val_length'] + time_params['test_length']))
        val_index = slice(-(time_params['val_length'] + time_params['test_length'] + time_params['header_length']),
                          -time_params['test_length'])
        test_index = slice(-(time_params['test_length'] + time_params['header_length']), None)

        train_series = target_time_series[train_index]  # 训练集
        val_series = target_time_series[val_index]  # 验证集
        test_series = target_time_series[test_index]  # 测试集

        # 使用 Darts 的 Scaler 对数据进行缩放
        if mode == 'training':
            train_scaler = Scaler(name='target').fit(train_series)  # 使用训练数据拟合缩放器
            past_cov_scaler = Scaler(name='covariate').fit(past_covariate_time_series)  # 使用过去的协变量拟合缩放器
            dump(train_scaler, open(get_data_file_path('scaler_train'), 'wb'))  # 保存缩放器
            dump(past_cov_scaler, open(get_data_file_path('scaler_past'), 'wb'))  # 保存缩放器
        elif mode == 'predicting':
            train_series = target_time_series  # 当执行预测任务时，不截取数据
            with open(get_data_file_path('scaler_train'), 'rb') as f:
                train_scaler = load(f)  # 加载缩放器
            with open(get_data_file_path('scaler_past'), 'rb') as f:
                past_cov_scaler = load(f)  # 加载缩放器
        else:
            raise ValueError("mode的值应为'training' 或 'predicting'")

        # 当目标值是0、1时， 不需要转换
        # train_scaled = train_scaler.transform(train_series)  # 转换训练数据
        train_scaled = train_series
        val_scaled = train_scaler.transform(val_series)  # 转换验证数据
        test_scaled = train_scaler.transform(test_series)  # 转换测试数据
        past_covariates_scaled = past_cov_scaler.transform(past_covariate_time_series)  # 转换过去的协变量

        # 使用 RBF 编码时间特征来生成未来的协变量
        latest_date = data_cleaned['date'].max()  # 获取最新的日期
        forecast_end_date = str(int(latest_date) + 10000)  # 设置预测的未来日期
        future_dates = xtdata.get_trading_calendar("SH", start_time=latest_date,
                                                   end_time=forecast_end_date)  # 获取未来的交易日历
        all_dates = np.concatenate((data_cleaned['date'].unique(), future_dates[1:]))  # 合并过去和未来的日期
        date_index = pd.DatetimeIndex(np.sort(all_dates))  # 创建日期时间索引

        future_encoded_features = rbf_encode_time_features(date_index)  # 使用 RBF 编码时间特征
        future_covariate_series = TimeSeries.from_dataframe(future_encoded_features).astype(np.float32)  # 未来协变量时间序列

        processed_series_data = {
            "train": train_scaled,  # 缩放后的训练数据
            "val": val_scaled,  # 缩放后的验证数据
            "test": test_scaled,  # 缩放后的测试数据
            "past_covariates": past_covariates_scaled,  # 缩放后的过去的协变量
            "future_covariates": future_covariate_series,  # 未来协变量
            "scaler_train": train_scaler,  # 目标变量的缩放器
            "scaler_past": past_cov_scaler  # 协变量的缩放器
        }

        # 保存处理后的数据
        for key, data in processed_series_data.items():
            data_path = get_data_file_path(key)  # 获取文件路径
            if key not in ['scaler_train', 'scaler_past']:
                data.to_pickle(data_path)  # 使用 pickle 保存
            else:
                dump(data, open(data_path, 'wb'))  # 使用 pickle 保存缩放器
            logger.info(f"数据 '{key}' 已保存到 {data_path}")  # 记录保存操作

        return processed_series_data

    except Exception as e:
        logger.critical(f"生成 processed_series_data 失败 - {str(e)}")  # 记录严重错误
        raise  # 重新抛出异常


def rbf_encode_time_features(dates: pd.DatetimeIndex, num_centers: int = 10) -> pd.DataFrame:
    """
    使用径向基函数 (RBF) 对时间特征进行编码。

    Args:
        dates (pd.DatetimeIndex): 日期时间索引。
        num_centers (int, optional): RBF 核函数的中心数量。 Defaults to 10.

    Returns:
        pd.DataFrame: 编码后的时间特征数据框。
    """
    day_scaler = MinMaxScaler()  # 用于日数据的 MinMaxScaler
    weekday_scaler = MinMaxScaler()  # 用于星期几数据的 MinMaxScaler
    month_scaler = MinMaxScaler()  # 用于月数据的 MinMaxScaler
    week_scaler = MinMaxScaler()  # 用于周数据的 MinMaxScaler

    days_scaled = day_scaler.fit_transform(dates.day.values.reshape(-1, 1)).flatten()  # 缩放日数据
    weekdays_scaled = weekday_scaler.fit_transform(dates.weekday.values.reshape(-1, 1)).flatten()  # 缩放星期几数据
    months_scaled = month_scaler.fit_transform(dates.month.values.reshape(-1, 1)).flatten()  # 缩放月数据
    weeks_scaled = week_scaler.fit_transform(dates.isocalendar().week.values.reshape(-1, 1)).flatten()  # 缩放周数据

    width = 1.0 / num_centers  # RBF 核函数的宽度
    centers = np.linspace(0, 1, num_centers)  # RBF 核函数的中心

    # 将 RBF 核应用于每个时间特征
    day_rbf = np.exp(-((days_scaled[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
    weekday_rbf = np.exp(-((weekdays_scaled[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
    month_rbf = np.exp(-((months_scaled[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
    week_rbf = np.exp(-((weeks_scaled[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))

    encoded_matrix = np.hstack([day_rbf, weekday_rbf, month_rbf, week_rbf])  # 合并编码后的特征
    return pd.DataFrame(encoded_matrix)  # 返回数据框


def get_data_file_path(key: str) -> Path:
    """
    获取数据存储的文件路径。

    Args:
        key (str): 数据键。

    Returns:
        Path: 文件路径。
    """
    return DATA_SAVE_PATHS[key]  # 从配置文件中返回文件路径


if __name__ == '__main__':
    # 执行主函数并存储结果
    result = generate_processed_series_data('training')  # 运行主函数
