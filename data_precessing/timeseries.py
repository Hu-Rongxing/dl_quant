import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from xtquant import xtdata
from pickle import dump, load
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# 自定义模块
from config import DATA_SAVE_PATHS, TIMESERIES_LENGTH
from utils.logger import logger
from .download_xt_data import get_data_from_local


def encode_and_scale_categorical_features(dataframe):
    """使用OrdinalEncoder和MinMaxScaler对分类特征进行编码和缩放。"""
    dataframe_scaled = dataframe.copy()
    categorical_cols = dataframe.select_dtypes(include=['object']).columns

    if not categorical_cols.empty:
        encoder = OrdinalEncoder()
        scaler = MinMaxScaler(feature_range=(0, 1))
        encoded_data = encoder.fit_transform(dataframe_scaled[categorical_cols])
        scaled_data = scaler.fit_transform(encoded_data)
        dataframe_scaled[categorical_cols] = scaled_data

    return dataframe_scaled


def fill_missing_values(dataframe):
    """通过前向和后向填充来填充缺失值。"""
    dataframe.sort_values(by='time', ascending=True, inplace=True)
    dataframe.filled = dataframe.ffill().bfill().reset_index(drop=True)
    dataframe.filled.fillna(0, inplace=True)
    return dataframe.filled


def add_time_index_sequence(dataframe):
    """为数据框添加顺序时间索引。"""
    dataframe.sort_values(by='time', ascending=True, inplace=True)
    dataframe_reset = dataframe.reset_index(drop=True)
    dataframe_reset['time'] = dataframe_reset.index
    dataframe_reset.index.name = 'time_seq'
    return dataframe_reset


def calculate_overnight_returns(dataframe):
    """计算并添加隔夜收益列到数据框。"""
    dataframe['overnight_return'] = dataframe['close'] / dataframe['preClose'] - 1
    dataframe.fillna(0, inplace=True)
    return dataframe


def generate_features(dataframe):
    """生成额外的滞后、滚动平均和技术指标特征。"""
    lags = [3, 5, 10, 20, 60]
    for lag in lags:
        for col in ['open', 'close', 'amount']:
            dataframe[f'{col}_lag_{lag}'] = dataframe[col].shift(lag)

    for col in ['open', 'close', 'amount']:
        dataframe[f'{col}_pct_change'] = dataframe[col].pct_change() * 100

    for col in ['open', 'high', 'low', 'close', 'amount']:
        dataframe[f'mean_{col}'] = dataframe[col].rolling(window=5).mean()

    dataframe['ma_7'] = dataframe['close'].rolling(window=7).mean()
    dataframe['ma_14'] = dataframe['close'].rolling(window=14).mean()
    dataframe['ma_21'] = dataframe['close'].rolling(window=21).mean()
    dataframe['ema_12'] = dataframe['close'].ewm(span=12, adjust=False).mean()
    dataframe['ema_26'] = dataframe['close'].ewm(span=26, adjust=False).mean()
    dataframe['ema_3'] = dataframe['overnight_return'].ewm(span=3, adjust=False).mean()
    dataframe['ema_5'] = dataframe['overnight_return'].ewm(span=5, adjust=False).mean()

    delta = dataframe['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    dataframe['rsi_14'] = 100 - (100 / (1 + rs))

    dataframe['rolling_max_high_14'] = dataframe['high'].rolling(window=14).max()
    dataframe['rolling_min_low_14'] = dataframe['low'].rolling(window=14).min()

    dataframe.fillna(method='bfill', inplace=True)
    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


def data_preprocessing(dataframe):
    """主要的数据清理函数，包括填充、排序和特征工程。"""
    dataframe.replace([np.inf, -np.inf], 0, inplace=True)
    dataframe = dataframe.reset_index(drop=False)
    dataframe = dataframe.groupby('stock_code').apply(fill_missing_values, include_groups=False)
    dataframe = dataframe.groupby('stock_code').apply(add_time_index_sequence, include_groups=False)
    dataframe = calculate_overnight_returns(dataframe)
    # dataframe['overnight_return'] = dataframe['overnight_return'].apply(lambda x: 1 if x > 0.002 else 0)
    dataframe = dataframe.groupby('stock_code').apply(generate_features, include_groups=False)
    dataframe.reset_index(inplace=True)
    dataframe.replace([np.inf, -np.inf], 0, inplace=True)
    return dataframe


def get_data_file_path(key: str) -> Path:
    """返回用于存储数据的文件路径。"""
    return DATA_SAVE_PATHS[key]


def load_and_clean_data() -> pd.DataFrame:
    """加载和清理原始数据。"""
    try:
        raw_data = get_data_from_local()
        clean_data = data_preprocessing(raw_data)
        clean_data['static_covariate'] = encode_and_scale_categorical_features(clean_data[['stock_code']])
        clean_data = fill_missing_values(clean_data)

        return clean_data
    except Exception as e:
        logger.error(f"加载或清理数据失败 - {str(e)}")
        raise


def create_combined_dataframe(clean_data: pd.DataFrame) -> tuple:
    """处理清理后的数据以创建目标和协变量数据框。"""
    feature_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'amount',
                      'overnight_return', 'open_lag_3', 'close_lag_3', 'amount_lag_3',
                      'open_lag_5', 'close_lag_5', 'amount_lag_5', 'open_lag_10',
                      'close_lag_10', 'amount_lag_10', 'open_lag_20', 'close_lag_20',
                      'amount_lag_20', 'open_lag_60', 'close_lag_60', 'amount_lag_60',
                      'open_pct_change', 'close_pct_change', 'amount_pct_change', 'mean_open',
                      'mean_high', 'mean_low', 'mean_close', 'mean_amount', 'ma_7', 'ma_14',
                      'ma_21', 'ema_12', 'ema_26', 'ema_3', 'ema_5', 'rsi_14',
                      'rolling_max_high_14', 'rolling_min_low_14']

    try:
        target_df = clean_data.pivot(index="time", columns="stock_code", values='overnight_return')
        target_df.replace([np.inf, -np.inf], 0, inplace=True)

        covariate_df = clean_data.pivot(index="time", columns="stock_code", values=feature_columns)
        covariate_df.columns.names = ['variable', 'stock_code']
        covariate_df.replace([np.inf, -np.inf], 0, inplace=True)

        logger.info("成功创建组合数据")
        return target_df, covariate_df
    except Exception as e:
        logger.error(f"创建组合数据失败 - {str(e)}")
        raise


def build_time_series(dataframe_tuple: tuple) -> tuple:
    """从数据框创建时间序列对象。"""
    try:
        target_time_series = TimeSeries.from_dataframe(dataframe_tuple[0]).astype(np.float32)
        past_covariate_time_series = TimeSeries.from_dataframe(dataframe_tuple[1]).astype(np.float32)
        logger.info("成功创建时间序列")
        return target_time_series, past_covariate_time_series
    except Exception as e:
        logger.error(f"创建时间序列失败 - {str(e)}")
        raise


def split_and_scaler_timeseries(target_time_series: TimeSeries, past_covariate_series: TimeSeries, mode: str = 'training') -> tuple:
    """为时间序列数据训练或加载缩放器。"""
    try:
        if mode not in ['training', 'predicting']:
            raise ValueError("模式必须是'training'或'predicting'。")

        time_params = TIMESERIES_LENGTH

        train_index = slice(time_params['header_length'], -(time_params['val_length'] + time_params['test_length']))
        val_index = slice(-(time_params['val_length'] + time_params['test_length'] + 30), -time_params['test_length'])
        test_index = slice(-(time_params['test_length'] + 30), None)

        train_series = target_time_series[train_index]
        val_series = target_time_series[val_index]
        test_series = target_time_series[test_index]

        if mode == 'training':
            train_scaler = Scaler().fit(train_series)
            past_cov_scaler = Scaler().fit(past_covariate_series)

            dump(train_scaler, open(get_data_file_path('scaler_train'), 'wb'))
            dump(past_cov_scaler, open(get_data_file_path('scaler_past'), 'wb'))
        else:
            with open(get_data_file_path('scaler_train'), 'rb') as f:
                train_scaler = load(f)
            with open(get_data_file_path('scaler_past'), 'rb') as f:
                past_cov_scaler = load(f)

        train_scaled = train_scaler.transform(train_series)
        val_scaled = train_scaler.transform(val_series)
        test_scaled = train_scaler.transform(test_series)
        past_covariate_scaled = past_cov_scaler.transform(past_covariate_series)

        logger.info(f"缩放器成功{'训练' if mode == 'training' else '加载'}")
        return train_scaled, val_scaled, test_scaled, past_covariate_scaled, train_scaler, past_cov_scaler
    except Exception as e:
        logger.error(f"缩放器准备失败 - {str(e)}")
        raise


def prepare_future_covariates(clean_data: pd.DataFrame) -> TimeSeries:
    """使用编码方案生成未来协变量数据。"""
    try:
        latest_date = clean_data['date'].max()
        forecast_end_date = str(int(latest_date) + 10000)  # 假设日期格式为YYYYMMDD
        future_dates = xtdata.get_trading_calendar("SH", start_time=latest_date, end_time=forecast_end_date)
        all_dates = np.concatenate((clean_data['date'].unique(), future_dates[1:]))
        date_index = pd.DatetimeIndex(np.sort(all_dates))
        future_encoded_features = rbf_encode_time_features(date_index)

        future_covariate_series = TimeSeries.from_dataframe(future_encoded_features).astype(np.float32)
        logger.info("成功生成未来协变量")
        return future_covariate_series
    except Exception as e:
        logger.error(f"生成未来协变量失败 - {str(e)}")
        raise


def rbf_function(x, centers, width):
    """计算时间编码的径向基函数（RBF）。"""
    return np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))


def rbf_encode_time_features(dates, num_centers=10):
    """使用RBF对时间特征（天、月）进行编码。"""
    day_scaler = MinMaxScaler()
    weekday_scaler = MinMaxScaler()
    month_scaler = MinMaxScaler()
    week_scaler = MinMaxScaler()

    days_scaled = day_scaler.fit_transform(dates.day.values.reshape(-1, 1)).flatten()
    weekdays_scaled = weekday_scaler.fit_transform(dates.weekday.values.reshape(-1, 1)).flatten()
    months_scaled = month_scaler.fit_transform(dates.month.values.reshape(-1, 1)).flatten()
    weeks_scaled = week_scaler.fit_transform(dates.isocalendar().week.values.reshape(-1, 1)).flatten()

    width = 1.0 / num_centers
    day_rbf = rbf_function(days_scaled, np.linspace(0, 1, num_centers), width)
    weekday_rbf = rbf_function(weekdays_scaled, np.linspace(0, 1, num_centers), width)
    month_rbf = rbf_function(months_scaled, np.linspace(0, 1, num_centers), width)
    week_rbf = rbf_function(weeks_scaled, np.linspace(0, 1, num_centers), width)

    encoded_matrix = np.hstack([day_rbf, weekday_rbf, month_rbf, week_rbf])
    return pd.DataFrame(encoded_matrix)


def prepare_timeseries_data(mode='training') -> dict:
    """处理并创建时间序列数据的主函数。"""
    try:
        if mode not in ['training', 'predicting']:
            raise ValueError("模式必须是'training'或'predicting'。")

        data_cleaned = load_and_clean_data()
        target_dataframe, covariate_dataframe = create_combined_dataframe(data_cleaned)
        target_timeseries, past_covariate_timeseries = build_time_series((target_dataframe, covariate_dataframe))

        train, val, test, past_covariates, scaler_train, scaler_past = split_and_scaler_timeseries(
            target_timeseries, past_covariate_timeseries, mode
        )

        future_covariate_timeseries = prepare_future_covariates(data_cleaned)

        processed_series_data = {
            "train": train,
            "val": val,
            "test": test,
            "past_covariates": past_covariates,
            "future_covariates": future_covariate_timeseries,
            "scaler_train": scaler_train,
            "scaler_past": scaler_past
        }

        for key, data in processed_series_data.items():
            data_path = get_data_file_path(key)
            if key not in ['scaler_train', 'scaler_past']:
                data.to_pickle(data_path)
            else:
                dump(data, open(data_path, 'wb'))
            logger.info(f"数据 {key} 已保存到 {data_path}")

        return processed_series_data

    except Exception as e:
        logger.critical(f"主流程执行失败 - {str(e)}")
        raise


if __name__ == '__main__':
    # 执行主函数并存储结果
    result = prepare_timeseries_data('training')