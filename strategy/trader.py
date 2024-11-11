from xtquant.xttype import StockAccount
from xtquant.xttrader import XtQuantTrader
from dotenv import load_dotenv
import os
import pathlib
import sys
import time
import math
from xtquant import xtconstant
from pathlib import Path
from datetime import datetime
# 自定义包
from utils.logger import logger
from config import config
from utils.others import generate_session_id
from utils.data import get_max_ask_price
from .qmt_callbacks import MyXtQuantTraderCallback
from .qmt_monitor import ProgramMonitor
from config.data_dic import order_type_dic


# 提权
os.environ.update({"__COMPAT_LAYER": "RunAsInvoker"})

# 导入自定义API
path = pathlib.Path(__file__).parent.parent
os.chdir(str(path.absolute()))
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

# 设置最大持仓数
MAX_POSITIONS = 3
load_dotenv()
acc = StockAccount(os.getenv("MINI_XT_USER"))
if not acc:
    raise RuntimeError('Account information could not be retrieved.')


def setup_xt_trader(acc=acc):
    callback = MyXtQuantTraderCallback()

    path = Path(config['xt_client']['program_dir']).parent.parent / 'userdata_mini/'
    session_id = generate_session_id()

    xt_trader = XtQuantTrader(str(path), session_id)
    xt_trader.register_callback(callback)
    xt_trader.start()

    # 尝试连接交易服务器
    connect_result = xt_trader.connect()
    if connect_result < 0:
        app = ProgramMonitor()
        app.restart_program()

        connect_result = xt_trader.connect()
        if connect_result < 0:
            raise RuntimeError('Failed to connect to XT')

    xt_trader.subscribe(acc)
    return xt_trader


try:
    xt_trader = setup_xt_trader()
except Exception as e:
    logger.critical("Critical error in main: ", exc_info=e)
    # xt_trader.subscribe(acc)
    xt_trader = setup_xt_trader()


def buy_stock_async(stocks, strategy_name='', order_remark=''):
    """
    买入股票函数：根据股票代码后缀确定所属市场并设置 order_type 后，异步发出买入指令。
    """

    logger.info("查询资产状况。")
    for i in range(15):
        xt_trader = setup_xt_trader()
        asset = xt_trader.query_stock_asset(acc)
        if asset is not None:
            break
        else:
            logger.warning(f"xt_trader.query_stock_asset返回值为None")
            time.sleep(1)

    if asset is None:
        logger.error(f"xt_trader.query_stock_asset返回值为None")

    cash = asset.cash
    logger.info(f"可用现金{cash}。")
    positions = xt_trader.query_stock_positions(acc)
    positions_stocks = [pos.stock_code for pos in positions]
    # 将结果转换回列表（如果需要）
    to_buy_stocks = [stock_code for stock_code in stocks if stock_code not in positions_stocks]
    position_list = [pos.stock_code for pos in positions if pos.volume > 0]
    position_count = len(position_list)
    available_slots = max(MAX_POSITIONS - position_count, 0)
    available_slots = min(available_slots, len(to_buy_stocks))
    logger.info(f"available_slots: {available_slots}")
    if available_slots == 0:
        logger.info(f"当前持仓已满:{position_list}。")
        return False

    # if len(stocks) > available_slots:
    #     stocks = stocks[:available_slots]

    # for stock in stocks:
    #     xtdata.subscribe_quote(stock, period="l2quote", count=-1)
    # time.sleep(2)

    for stock_code in to_buy_stocks:
        if stock_code.endswith('.SH') or stock_code.endswith('.SZ'):
            order_type = xtconstant.FIX_PRICE
        else:
            order_type = xtconstant.FIX_PRICE

        logger.info(f"股票【{stock_code}】报价类型为：{order_type}")

        # 读取最高要价
        max_ask_price = get_max_ask_price(stock_code)

        if max_ask_price == 999999:
            logger.warning(f"股票已经涨停：{stock_code}")
            continue

        if max_ask_price == 999998:
            logger.warning(f"当前合约不可交易：{stock_code}")
            continue

        if not max_ask_price:
            logger.warning(f"未能获得股票数据：{stock_code}")
            continue

        if max_ask_price == 0:
            logger.warning(f"委卖价为0，请检查{stock_code}的数据。")
            continue

        quantity = math.floor(cash / max_ask_price / available_slots / 100) * 100
        if quantity < 100:
            logger.info(f"{stock_code} 可买数量不足，现金：{cash}, 当前股价：{max_ask_price}")
            continue

        # response = xt_trader.order_stock_async(acc, stock_code, xtconstant.STOCK_BUY, quantity, order_type,
        #                                        max_ask_price,
        #                                        strategy_name, order_remark)
        response = xt_trader.order_stock(
            account=acc,
            stock_code=stock_code,
            order_type=xtconstant.STOCK_BUY,
            order_volume=quantity,
            price_type=order_type,
            price=max_ask_price,
            strategy_name=strategy_name,
            order_remark=order_remark
        )
        logger.info("完成提交。")
        if response < 0:
            logger.trader(
                f'\n【提交下单失败！- 买入 - {strategy_name}】\n 股票【{stock_code}】，\n数量【{quantity}】，\n单价【{max_ask_price}】，\n金额【{quantity * max_ask_price}】，\n返回值【{response}】')
        else:
            logger.trader(
                f'\n【提交下单成功！- 买入 - {strategy_name}】\n 股票【{stock_code}】，\n数量【{quantity}】，\n单价【{max_ask_price}】，\n金额【{quantity * max_ask_price}】，\n返回值【{response}】')


def generate_trading_report():
    # order_type_dic = {23: "买入", 24: "卖出"}

    today = datetime.now().strftime("%Y-%m-%d")

    # 查询资产
    asset = xt_trader.query_stock_asset(account=acc)

    # 查询持仓
    positions = xt_trader.query_stock_positions(account=acc)

    # 查询当天成交记录
    trades = xt_trader.query_stock_trades(account=acc)

    # 生成报告

    report = f"\n\n交易报告 - {today}\n\n"
    report += "=" * 20

    report += f"\n资产情况:\n"
    report += f"--资产代码: {asset.account_id}， 总资产: {asset.total_asset}， 股票市值: {asset.market_value}， 可用现金: {asset.cash}\n"

    report += "=" * 20

    report += "\n持仓情况:\n"
    for position in positions:
        report += f"--股票代码: {position.stock_code}， 股票市值: {position.market_value}， 持股数量: {position.volume}， 平均成本: {position.avg_price}\n"

    report += "=" * 20

    report += "\n当日成交:\n"
    for trade in trades:
        traded_time = datetime.fromtimestamp(trade.traded_time)
        order_type = order_type_dic.get(trade.order_type, '未定义')
        report += f"--{order_type}-{trade.strategy_name}】股票代码: {trade.stock_code}， 成交金额: {trade.traded_amount}， 成交数量: {trade.traded_volume}， 成交价格: {trade.traded_price}， 成交时间: {traded_time}， 备注：{trade.order_remark}\n"

    report += "=" * 20

    logger.trader(report)

    return report