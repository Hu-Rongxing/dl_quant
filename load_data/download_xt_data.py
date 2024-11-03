import os
from datetime import datetime
import pandas as pd
import xtquant.xtdata as xtdata
from typing import List, Optional
# 自定义
from strategy.qmt_monitor import start_xt_client
from utils.data import get_targets_list_from_csv
from utils.logger import logger

def download_history_data(
        stock_list: Optional[List[str]] = None,
        period: str = '1d',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        callback: Optional[callable] = None,
        incrementally: bool = False
) -> None:
    """
    下载指定股票列表的历史数据。

    :param stock_list: 股票代码列表，默认为从 CSV 文件获取
    :param period: 时间周期，默认 '1d'
    :param start_time: 起始时间，格式为 'YYYYMMDD'，默认 '20160101'
    :param end_time: 结束时间，格式为 'YYYYMMDD%H%M%S'，默认当前日期
    :param callback: 下载数据时的回调函数，默认 None
    :param incrementally: 是否增量下载，默认 False
    """
    if stock_list is None:
        stock_list = get_targets_list_from_csv()
    start_time = start_time or '20160101'
    end_time = end_time or datetime.now().strftime('%Y%m%d%H%M%S')

    for stock in stock_list:
        try:
            xtdata.download_history_data(stock, period, start_time, end_time, incrementally=incrementally)
            logger.info(f"成功下载股票数据：{stock}")
        except Exception as e:
            logger.error(f"下载股票数据失败：{stock}，错误信息：{e}")
            start_xt_client()



def get_data_from_local(
        period: str = '1d',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        stock_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    获取历史股票数据并返回为 pandas DataFrame。

    :param period: 时间周期，默认 '1d'
    :param start_time: 起始时间，格式为 'YYYYMMDD'，默认 '20160101'
    :param end_time: 结束时间，格式为 'YYYYMMDD%H%M%S'，默认当前日期
    :param stock_list: 股票列表，格式为 ['xxxxxx.XX','xxxxxx.XX']，默认为None
    :return: 包含股票数据的 pandas DataFrame
    """
    start_time = start_time or '20160101'
    end_time = end_time or datetime.now().strftime('%Y%m%d%H%M%S')
    if not stock_list:
        stock_list = get_targets_list_from_csv()

    # 下载数据
    download_history_data(stock_list=stock_list, period=period, start_time=start_time, end_time=end_time,
                          incrementally=False)

    try:
        market_data = xtdata.get_local_data(
            field_list=[],
            stock_list=stock_list,
            period=period,
            start_time=start_time,
            end_time=end_time,
            count=-1,
            dividend_type='front',
            fill_data=True
        )

        data_df = pd.concat(
            [df.assign(stock_code=field).rename_axis('date') for field, df in market_data.items()],
            ignore_index=False
        )
        return data_df
    except Exception as e:
        logger.error(f"获取股票数据失败，错误信息：{e}")
        return pd.DataFrame()  # 返回空的 DataFrame 以防止后续代码崩溃
