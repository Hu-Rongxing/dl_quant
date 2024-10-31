"""
  从迅投客户端（strategy)加载数据。加载的数据包括三类：
  1. 市场数据，包括股票数据等，返回的格式为pandas.DataFrame
  2. 用于darts模型的数据，返回的格式为darts.Timeseries
"""

# 读取市场数据，返回格式为pandas.DataFrame
from .download_xt_data import get_data_from_local, download_history_data
from .multivariate_timeseries import generate_processed_series_data

__all__ = [get_data_from_local, download_xt_data, generate_processed_series_data]
