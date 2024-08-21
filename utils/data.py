from pathlib import Path
import csv
from typing import List
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
