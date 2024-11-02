import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from pickle import dump, load
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from typing import Tuple, Dict, List, Optional
from scipy.signal import argrelextrema

# 自定义模块
from config import DATA_SAVE_PATHS
from utils.logger import logger
from load_data.download_xt_data import get_data_from_local

# 1. 获取数据
price_data_df = get_data_from_local().reset_index()
# 生成日期到索引的映射，日期相同的映射到相同的索引
unique_dates = sorted(pd.to_datetime(price_data_df['date'].unique()))
date_to_index_mapper = {date: idx for idx, date in enumerate(unique_dates)}


# price_data_df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount',
#        'settelementPrice', 'openInterest', 'preClose', 'suspendFlag',
#        'stock_code']

# 2. 清洗数据
# 添加整数索引
def add_int_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    为数据添加整数索引, 日期与索引保持一致。
    """
    # 确保 'date' 列为日期类型
    df['date_index'] = pd.to_datetime(df['date'])
    # 在原 DataFrame 中添加 'index' 列，映射日期到索引
    df['date_index'] = df['date_index'].map(date_to_index_mapper)
    return df


price_data_df = add_int_index(price_data_df)
# 删除开始的缺失值，保留后面的缺失值，保持date_index的连续性
def drop_first_continue_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='date_index')
    df = df.reset_index(inplace=False, drop=True)
    # 找到第一个不含任何 NaN 值的行的索引
    first_valid_index = df['open'].first_valid_index()
    # 从第一个有效行开始，保留后面的所有行
    df_cleaned = df.loc[first_valid_index: , : ].reset_index(drop=True)
    logger.debug(f"删除从第 {first_valid_index} 行开始含有 NaN 的连续行。")

    return df_cleaned


logger.debug(f"数据长度{len(price_data_df)}行")

def calculate_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    计算并添加目标列到数据框。
    当未来三天内，最大涨幅超过1%，且最小涨幅大于-0.1%时，target列的值为1，否则为0。

    Args:
        dataframe (pd.DataFrame): 输入数据框。

    Returns:
        pd.DataFrame: 添加目标列后的数据框。
    """
    # 检查是否包含必要的列
    if 'high' not in dataframe.columns or 'low' not in dataframe.columns or 'close' not in dataframe.columns:
        raise ValueError("DataFrame 中缺少必要的列：'high', 'low', 'close'。")

    # 初始化目标列
    dataframe['target'] = 0

    # 计算未来三天的最高价和最低价
    future_highs = dataframe['high'].shift(-1).rolling(window=3).max()
    future_lows = dataframe['low'].shift(-1).rolling(window=3).min()
    future_closes = dataframe['close']

    # 计算最大涨幅和最小跌幅
    max_increase = (future_highs - future_closes) / future_closes
    min_drop = (future_lows - future_closes) / future_closes

    # 条件：最大涨幅超过1%且最小跌幅大于-0.1%
    threshold = 0.01
    mask = (max_increase > threshold) & (min_drop > -1 * threshold)

    # 根据条件更新目标列
    dataframe.loc[mask, 'target'] = 1

    # 填充缺失值
    dataframe.fillna(0, inplace=True)

    return dataframe

def apply_mapper(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = drop_first_continue_rows(dataframe)
    df = calculate_target(df)
    return df

price_data_df_2 = price_data_df.groupby('stock_code', group_keys=True).apply(
    apply_mapper, include_groups=False
).reset_index(level='stock_code', drop=False)

print(f"{sum(price_data_df_2['target'] == 1)/len(price_data_df_2['target']):.2%}")


list_of_target  = []
# 将数据转换为时间序列格式，并按商店和家庭进行分组
list_of_target = TimeSeries.from_group_dataframe(
    df=price_data_df,
    time_col="date_index",
    group_cols=["stock_code"],  # 按 `group_cols` 分组提取时间序列
    static_cols=None,
    value_cols="target",  # 关注的协变量，即促销情况
    fill_missing_dates=True,
    freq=1  # 设置频率为每天
)

# 将时间序列数据转换为 float32 类型
for ts in list_of_TS_promo:
    ts = ts.astype(np.float32)

list_of_past_covariates = []
list_of_future_covariates = []
# 可逆数据转换器

# 数据mappter

def encode_and_scale_categorical_features(
        dataframe: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    使用 OrdinalEncoder 和 MinMaxScaler 对分类特征进行编码和缩放。

    Args:
        dataframe (pd.DataFrame): 输入数据框。
        categorical_cols (Optional[List[str]]): 分类特征列的列表。如果为 None，默认为空列表。

    Returns:
        pd.DataFrame: 编码和缩放后的数据框。
    """
    if categorical_cols is None:
        categorical_cols = []

    dataframe_scaled = dataframe.copy()

    if categorical_cols:
        encoder = OrdinalEncoder()
        scaler = MinMaxScaler(feature_range=(0, 1))
        encoded_data = encoder.fit_transform(dataframe_scaled[categorical_cols])
        scaled_data = scaler.fit_transform(encoded_data)
        dataframe_scaled[categorical_cols] = scaled_data

    return dataframe_scaled


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    填充缺失值。

    Args:
        dataframe (pd.DataFrame): 输入数据框。

    Returns:
        pd.DataFrame: 填充后的数据框。
    """
    dataframe.sort_values(by='time', ascending=True, inplace=True)
    dataframe_filled = dataframe.ffill().reset_index(drop=True)
    dataframe_filled.fillna(0, inplace=True)
    return dataframe_filled


def add_time_index_sequence(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    为数据框添加顺序时间索引。

    Args:
        dataframe (pd.DataFrame): 输入数据框。

    Returns:
        pd.DataFrame: 添加时间索引后的数据框。
    """
    dataframe.sort_values(by='time', ascending=True, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


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





def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    生成额外的滞后、滚动平均和技术指标特征。

    Args:
        dataframe (pd.DataFrame): 输入数据框。

    Returns:
        pd.DataFrame: 添加特征后的数据框。
    """
    dataframe['ma_7'] = dataframe['close'].rolling(window=7).mean()
    dataframe['ma_14'] = dataframe['close'].rolling(window=14).mean()
    dataframe['ma_21'] = dataframe['close'].rolling(window=21).mean()
    dataframe['ema_12'] = dataframe['close'].ewm(span=12, adjust=False).mean()
    dataframe['ema_26'] = dataframe['close'].ewm(span=26, adjust=False).mean()

    delta = dataframe['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    dataframe['rsi_14'] = 100 - (100 / (1 + rs))

    dataframe['rolling_max_high_14'] = dataframe['high'].rolling(window=14).max()
    dataframe['rolling_min_low_14'] = dataframe['low'].rolling(window=14).min()

    dataframe.bfill(inplace=True)
    return dataframe


def data_preprocessing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    主要的数据清理函数，包括填充、排序和特征工程。

    Args:
        dataframe (pd.DataFrame): 输入数据框。

    Returns:
        pd.DataFrame: 处理后的数据框。
    """
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    dataframe = dataframe.groupby('stock_code', group_keys=False).apply(fill_missing_values)
    dataframe = dataframe.groupby('stock_code', group_keys=False).apply(add_time_index_sequence)
    dataframe.reset_index(drop=True, inplace=True)

    dataframe = calculate_target(dataframe)
    # dataframe['target'] = dataframe.groupby('stock_code')['target'].shift(-1)
    dataframe = dataframe.groupby('stock_code', group_keys=False).apply(generate_features)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe.fillna(0, inplace=True)
    return dataframe


def get_data_file_path(key: str) -> Path:
    """
    返回用于存储数据的文件路径。

    Args:
        key (str): 数据类型的键。

    Returns:
        Path: 数据存储路径。
    """
    return Path(f"data/scalers/{key}")


def load_and_clean_data() -> pd.DataFrame:
    """
    加载和清理原始数据。

    Returns:
        pd.DataFrame: 清理后的数据框。
    """
    try:
        raw_data = get_data_from_local()
        clean_data = data_preprocessing(raw_data)
        clean_data['code'] = clean_data['stock_code']
        clean_data = encode_and_scale_categorical_features(clean_data, ['code'])
        clean_data = fill_missing_values(clean_data)
        return clean_data
    except Exception as e:
        logger.error(f"加载或清理数据失败 - {str(e)}")
        raise e


def create_time_series_dict(
        clean_data: pd.DataFrame
) -> Tuple[Dict[str, TimeSeries], Dict[str, TimeSeries]]:
    """
    为每个股票代码创建单独的时间序列对象，使用整数索引。

    Args:
        clean_data (pd.DataFrame): 清理后的数据框。

    Returns:
        Tuple[Dict[str, TimeSeries], Dict[str, TimeSeries]]: 目标和协变量时间序列的字典。
    """
    try:
        target_series_dict = {}
        covariate_series_dict = {}

        feature_columns = [
            'open', 'high', 'low', 'close', 'amount',
            'ma_7', 'ma_14', 'ma_21', 'ema_12', 'ema_26',
            'rsi_14', 'rolling_max_high_14',
            'rolling_min_low_14'
        ]

        stocks = clean_data['stock_code'].unique()

        for stock in stocks:
            stock_data = clean_data[clean_data['stock_code'] == stock].reset_index(drop=True)

            # 创建目标时间序列
            target_series = TimeSeries.from_dataframe(
                stock_data[['c0', 'c1']].astype(np.float32),
                fill_missing_dates=True,
                freq=None
            )

            # 创建协变量时间序列
            covariate_series = TimeSeries.from_dataframe(
                stock_data[feature_columns].astype(np.float32),
                fill_missing_dates=True,
                # freq=None
            )

            target_series_dict[stock] = target_series
            covariate_series_dict[stock] = covariate_series

        logger.info("成功为每个股票创建时间序列")
        return target_series_dict, covariate_series_dict

    except Exception as e:
        logger.error(f"创建时间序列失败 - {str(e)}")
        raise e


def split_and_scale_time_series(
        series_dict: Dict[str, TimeSeries],
        mode: str = 'training'
) -> Tuple[Dict[str, Dict[str, TimeSeries]], Dict[str, Scaler]]:
    """
    为每个股票的时间序列进行训练、验证、测试拆分和缩放。

    Args:
        series_dict (Dict[str, TimeSeries]): 时间序列的字典。
        mode (str): 模式，'training'或'predicting'。

    Returns:
        Tuple[Dict[str, Dict[str, TimeSeries]], Dict[str, Scaler]]: 包含拆分和缩放后时间序列的字典，以及对应的缩放器。
    """
    try:
        scaled_series = {}
        scalers = {}

        for stock, series in series_dict.items():
            # 创建缩放器并拟合
            scaler = Scaler()
            scaler.fit(series)
            scalers[stock] = scaler

            # 缩放序列
            scaled_series_full = scaler.transform(series)

            total_length = len(series)
            train_end = int(total_length * 0.6)
            val_end = int(total_length * 0.8)

            train_series = scaled_series_full[:train_end]
            val_series = scaled_series_full[train_end:val_end]
            test_series = scaled_series_full[val_end:]

            scaled_series[stock] = {
                'train': train_series,
                'val': val_series,
                'test': test_series
            }

        logger.info(f"时间序列成功{'训练' if mode == 'training' else '加载'}和缩放")
        return scaled_series, scalers

    except Exception as e:
        logger.error(f"缩放时间序列失败 - {str(e)}")
        raise e


def scale_time_series(
        series_dict: Dict[str, TimeSeries]
) -> Tuple[Dict[str, TimeSeries], Dict[str, Scaler]]:
    """
    对每个股票的协变量时间序列进行缩放，不拆分。

    Args:
        series_dict (Dict[str, TimeSeries]): 协变量时间序列的字典。

    Returns:
        Tuple[Dict[str, TimeSeries], Dict[str, Scaler]]: 缩放后的时间序列字典和对应的缩放器。
    """
    try:
        scaled_series = {}
        scalers = {}

        for stock, series in series_dict.items():
            scaler = Scaler()
            scaler.fit(series)
            scalers[stock] = scaler

            scaled_series[stock] = scaler.transform(series)

        logger.info("协变量时间序列成功缩放")
        return scaled_series, scalers

    except Exception as e:
        logger.error(f"缩放协变量时间序列失败 - {str(e)}")
        raise e


def prepare_future_covariates(clean_data: pd.DataFrame) -> Dict[str, TimeSeries]:
    """
    为每个股票生成未来协变量时间序列。

    Args:
        clean_data (pd.DataFrame): 清理后的数据框。

    Returns:
        Dict[str, TimeSeries]: 未来协变量时间序列的字典。
    """
    try:
        future_covariate_series_dict = {}
        stocks = clean_data['stock_code'].unique()

        for stock in stocks:
            stock_data = clean_data[clean_data['stock_code'] == stock]
            date_index = pd.RangeIndex(start=0, stop=len(stock_data), step=1)

            # 对日期进行 RBF 编码
            future_encoded_features = rbf_encode_time_features(date_index)

            future_covariate_series = TimeSeries.from_dataframe(
                future_encoded_features.astype(np.float32),
                fill_missing_dates=True,
                freq=None
            )
            future_covariate_series_dict[stock] = future_covariate_series

        logger.info("成功为每个股票生成未来协变量")
        return future_covariate_series_dict
    except Exception as e:
        logger.error(f"生成未来协变量失败 - {str(e)}")
        raise e


def rbf_function(
        x: np.ndarray,
        centers: np.ndarray,
        width: float
) -> np.ndarray:
    """
    计算时间编码的径向基函数（RBF）。

    Args:
        x (np.ndarray): 输入数据。
        centers (np.ndarray): RBF 中心。
        width (float): RBF 宽度。

    Returns:
        np.ndarray: RBF 计算结果。
    """
    return np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))


def rbf_encode_time_features(
        indices: pd.Index,
        num_centers: int = 10
) -> pd.DataFrame:
    """
    使用 RBF 对整数索引进行编码。

    Args:
        indices (pd.Index): 整数索引。
        num_centers (int): RBF 中心数量。

    Returns:
        pd.DataFrame: 编码后的时间特征数据框。
    """
    index_values = indices.values.astype(float)
    scaler = MinMaxScaler()
    indices_scaled = scaler.fit_transform(index_values.reshape(-1, 1)).flatten()

    width = 1.0 / num_centers
    centers = np.linspace(0, 1, num_centers)
    rbf_encoded = rbf_function(indices_scaled, centers, width)

    return pd.DataFrame(rbf_encoded, index=indices)


def prepare_timeseries_data(mode: str = 'training') -> Dict[str, Dict[str, TimeSeries]]:
    """
    处理并创建时间序列数据的主函数。

    Args:
        mode (str): 模式，'training'或'predicting'。

    Returns:
        Dict[str, Dict[str, TimeSeries]]: 处理后的时间序列数据字典。
    """
    try:
        if mode not in ['training', 'predicting']:
            raise ValueError("模式必须是 'training' 或 'predicting'。")

        data_cleaned = load_and_clean_data()
        target_series_dict, covariate_series_dict = create_time_series_dict(data_cleaned)

        # 对目标时间序列进行拆分和缩放
        target_scaled, target_scalers = split_and_scale_time_series(target_series_dict, mode)

        # 对协变量时间序列只进行缩放，不拆分
        covariate_scaled, covariate_scalers = scale_time_series(covariate_series_dict)

        # 准备未来协变量，并进行缩放
        future_covariate_series_dict = prepare_future_covariates(data_cleaned)
        future_covariate_scaled, future_covariate_scalers = scale_time_series(future_covariate_series_dict)

        # 将所有数据整合到一起
        processed_series_data = {}
        for stock in target_series_dict.keys():
            processed_series_data[stock] = {
                "train": target_scaled[stock]['train'],
                "val": target_scaled[stock]['val'],
                "test": target_scaled[stock]['test'],
                "past_covariates": covariate_scaled[stock],
                "future_covariates": future_covariate_scaled[stock],
            }

        # 保存数据和缩放器
        for stock, data in processed_series_data.items():
            for key, series in data.items():
                data_path = get_data_file_path(f'{stock}_{key}')
                series.to_pickle(data_path)
                logger.info(f"数据 {stock} - {key} 已保存到 {data_path}")

            # 保存缩放器
            scaler_path = get_data_file_path(f'{stock}_target_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                dump(target_scalers[stock], f)
                logger.info(f"目标缩放器 {stock} 已保存到 {scaler_path}")

            covariate_scaler_path = get_data_file_path(f'{stock}_covariate_scaler.pkl')
            with open(covariate_scaler_path, 'wb') as f:
                dump(covariate_scalers[stock], f)
                logger.info(f"协变量缩放器 {stock} 已保存到 {covariate_scaler_path}")

            future_covariate_scaler_path = get_data_file_path(f'{stock}_future_covariate_scaler.pkl')
            with open(future_covariate_scaler_path, 'wb') as f:
                dump(future_covariate_scalers[stock], f)
                logger.info(f"未来协变量缩放器 {stock} 已保存到 {future_covariate_scaler_path}")

        return processed_series_data

    except Exception as e:
        logger.critical(f"主流程执行失败 - {str(e)}")
        raise e


if __name__ == '__main__':
    # 执行主函数并存储结果
    result = prepare_timeseries_data('training')
    pass
