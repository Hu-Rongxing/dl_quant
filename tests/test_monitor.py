# 导入您的函数
from load_data import download_history_data, generate_processed_series_data
from utils.others import is_trading_day, is_transaction_hour
from strategy.stop_loss import stop_loss_main
from models.TFTModel_dep import fit_model, predict_market
from strategy.trader import buy_stock_async
from strategy.qmt_monitor import start_xt_client
from strategy.trader import generate_trading_report
from utils.logger import logger


def test_monitor_functions():
    start_xt_client()
    generate_trading_report()
    download_history_data()
    generate_processed_series_data()
    is_trading_day()
    is_transaction_hour()

def test_stop_loss_main():
    stop_loss_main()

def test_models():
    fit_model()

def test_predict_market():
    predict_market()

def test_buy_stock_async():
    from strategy.trader import buy_stock_async
    buy_stock_async(["999999.SH", "000001.SZ"],strategy_name='买入策略', order_remark='策略买入。')

