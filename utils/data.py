import csv
from xtquant import xtdata
from datetime import datetime
from pathlib import Path
import math
from typing import List, Optional
# 自定义
from utils.others import is_trading_day
from .logger import logger


def get_targets_list_from_csv() -> List[str]:
    """
    从 CSV 文件中读取有效的股票代码列表。

    :return: 返回一个包含有效股票代码的列表。
    """
    # 定义 CSV 文件路径
    csv_file_path = Path(__file__).parent.parent / "data/investment_target/investment_targets.csv"
    stock_list: List[str] = []  # 初始化股票代码列表

    try:
        # 打开 CSV 文件并读取内容
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # 遍历每一行，检查 STATUS 列是否为 True
            for row in reader:
                if row['STATUS'] == 'True':
                    stock_list.append(row['SECURE'])
    except Exception as e:
        # 如果读取文件出错，记录错误日志
        logger.error(f"读取 CSV 文件出错: {e}")

    return stock_list


def get_max_ask_price(stock_code):
    """
    获取指定股票代码的五档行情最高报价

    :param stock_code: 股票代码，例如 "000001.SZ"
    :return: 最新股价，如果获取失败则返回 None
    """
    try:
        # 确保 MiniQmt 已经有所需的数据
        code_list = [stock_code]
        data = xtdata.get_full_tick(code_list)

        if stock_code in data and bool(data[stock_code]):
            time = data[stock_code]['timetag']
            max_ask_price = max(
                max(data[stock_code]['askPrice']),  # 最高卖价
                max(data[stock_code]['bidPrice']),  # 最高买价
                data[stock_code]['lastPrice'] * 1.01  # 最新价+1%
            )
            max_ask_price = math.ceil(max_ask_price * 100) / 100
            instrument = xtdata.get_instrument_detail(stock_code)
            # 成交价等于涨停价时
            if data[stock_code]['lastPrice'] == instrument["UpStopPrice"]:
                logger.warning(f"{stock_code}涨停")
                return 999999
            # 不超过涨停价
            if instrument["UpStopPrice"] > 0:
                max_ask_price = min(max_ask_price, instrument["UpStopPrice"])
            else:
                logger.warning(f"{stock_code}涨停价异常")

            # max_ask_price = round(max_ask_price, 2)
            # 信息
            logger.info(f"股票:{stock_code}; 时间:{time}; 价格:{max_ask_price}")
            return max_ask_price
        else:
            logger.error(f"未能获取股票 {stock_code} 的数据")
            return None
    except Exception as e:
        logger.error(f"获取股票 {stock_code} 的数据时发生错误: {e}")
        return None


def on_subscribe_data(datas):
    for stock_code in datas:
        print(stock_code, datas[stock_code])


def subscribe_whole_real_data(test=False):
    """订阅全推行情数据"""
    if not (is_trading_day or test):
        logger.info("今天不是交易日")
        return False
    res = xtdata.connect()
    xtdata.subscribe_whole_quote(["SH", "SZ"])
    logger.info(f"订阅实时行情:{res}")
    xtdata.run()
    return xtdata


# def subscribe_real_data(period="1d", test=False):
def subscribe_real_data(period="1d", test=False):
    """订阅全推行情数据"""
    if not (is_trading_day or test):
        logger.info("今天不是交易日")
        return False
    res = xtdata.connect()
    stock_list = get_targets_list_from_csv()
    # 向服务器订阅数据
    for stock in stock_list:
        xtdata.subscribe_quote(stock, period=period, count=-1)  # 设置count = -1来取到当天所有实时行情
        logger.info(f"订阅全推行情:{stock}")
    xtdata.run()


def unsubscribe_real_data(period='1d', test=False):
    """订阅全推行情数据"""
    if not (is_trading_day or test):
        logger.info("今天不是交易日")
        return False
    res = xtdata.connect()
    stock_list = get_targets_list_from_csv()
    # 向服务器订阅数据
    for stock in stock_list:
        xtdata.unsubscribe_quote(stock, period=period, count=-1)  # 设置count = -1来取到当天所有实时行情
        logger.info(f"取消订阅全推行情:{stock}")
    # print(stock_list)
    # xtdata.subscribe_quote(stock_list)
    xtdata.run()



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


def identify_security_type(code):
    # 提取证券代码的基础部分
    code_base = code.split('.')[0]

    # 判断证券类型
    if code_base.startswith(('512', '513', '515', '516', '518', '588', '159', '161')):
        return "ETF"
    elif code_base.startswith(('600', '601', '000', '002', '300')):
        return "普通股票"
    elif code_base.startswith(('110', '113', '111', '112')):
        return "债券"

    return "未知类型"


if __name__ == '__main__':
    subscribe_real_data(test=False)
    # get_max_ask_price("000001.SZ")
