import time
import multiprocessing
import functools
from apscheduler.schedulers.background import BackgroundScheduler

# 导入您的函数  
from load_data import download_history_data
from utils.others import is_trading_day, is_transaction_hour
from strategy.stop_loss import stop_loss_main
from models.TFTModel_dep import fit_model, predict_market
from strategy.trader import buy_stock_async
from strategy.qmt_monitor import start_xt_client
from strategy.trader import generate_trading_report
from utils.logger import logger


def buying_strategy():
    """  
    买入策略函数，判断是否为交易日和交易时间段，然后预测市场并执行买入操作。  
    """
    if not is_trading_day():
        logger.info("今天不是交易日")
        return
    if not is_transaction_hour():
        logger.info("当前不在交易时间段内")
        return
    to_buy = predict_market()
    if to_buy:
        logger.trader(f"买入股票列表：{'、'.join(to_buy)}")
        buy_stock_async(to_buy, strategy_name='买入策略', order_remark='策略买入。')
    else:
        logger.info("无股票可买入。")


def retry(max_attempts=3, wait_seconds=1):
    """  
    函数重试装饰器，在函数抛出异常时重试指定次数。  

    Args:  
        max_attempts (int): 最大尝试次数。  
        wait_seconds (int): 每次重试前等待的秒数。  
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.error(
                        f"函数 {func.__name__} 执行失败。第 {attempts} 次尝试，共 {max_attempts} 次。错误信息：{e}")
                    if attempts >= max_attempts:
                        logger.error(f"函数 {func.__name__} 在重试 {max_attempts} 次后仍然失败。")
                        raise
                    time.sleep(wait_seconds)

        return wrapper

    return decorator


@retry(max_attempts=3)
def download_history_data_task():
    """  
    下载历史数据的任务函数。  
    """
    if is_trading_day():
        download_history_data()
    else:
        logger.info("今天不是交易日，跳过下载历史数据。")


@retry(max_attempts=3)
def fit_model_task():
    """  
    训练模型的任务函数。  
    """
    if is_trading_day():
        fit_model()
    else:
        logger.info("今天不是交易日，跳过模型训练。")


@retry(max_attempts=3)
def buying_strategy_task():
    """  
    买入策略的任务函数。  
    """
    if is_trading_day():
        buying_strategy()
    else:
        logger.info("今天不是交易日，跳过买入策略。")


@retry(max_attempts=3)
def generate_trading_report_task():
    """  
    生成交易报告的任务函数。  
    """
    if is_trading_day():
        generate_trading_report()
    else:
        logger.info("今天不是交易日，跳过生成交易报告。")


def is_stop_loss_running():
    """  
    检查 stop_loss_main 是否已经在运行。  
    """
    for process in multiprocessing.active_children():
        if process.name == 'stop_loss_main_process':
            return True
    return False


def run_stop_loss():
    """  
    运行 stop_loss_main 的函数，用于多进程启动。  
    """
    try:
        stop_loss_main()
    except Exception as e:
        logger.error(f"stop_loss_main 运行中发生错误：{e}")


def stop_loss_main_task():
    """  
    止损策略任务函数，检查并启动止损进程。  
    """
    if not is_trading_day():
        logger.info("今天不是交易日，跳过止损策略。")
        return

    if not is_stop_loss_running():
        p = multiprocessing.Process(target=run_stop_loss, name='stop_loss_main_process')
        p.start()
        logger.info("已经启动 stop_loss_main 进程。")
    else:
        logger.info("stop_loss_main 已经在运行中。")


@retry(max_attempts=3)
def start_xt_client_task():
    """  
    启动交易客户端的任务函数。  
    """
    if is_trading_day():
        start_xt_client()
    else:
        logger.info("今天不是交易日，跳过启动交易客户端。")


if __name__ == '__main__':
    # 创建调度器  
    scheduler = BackgroundScheduler()

    # 添加任务  

    # 1. 每个交易日13:00、16:00执行 download_history_data。  
    scheduler.add_job(download_history_data_task, 'cron', day_of_week='mon-fri', hour='13,16', minute='0',
                      id='download_history_data_task_13_16')

    # 2. 每个交易日9:00先执行 download_history_data_task，再执行 fit_model_task。  
    # 先在9:00执行download_history_data_task  
    scheduler.add_job(download_history_data_task, 'cron', day_of_week='mon-fri', hour='9', minute='0',
                      id='download_history_data_task_9')
    # 然后在9:05执行fit_model_task，确保数据下载完成  
    scheduler.add_job(fit_model_task, 'cron', day_of_week='mon-fri', hour='9', minute='5', id='fit_model_task')

    # 3. 每个交易日14:57先执行 download_history_data_task，再执行 buying_strategy_task。  
    # 先在14:57执行download_history_data_task  
    scheduler.add_job(download_history_data_task, 'cron', day_of_week='mon-fri', hour='14', minute='57',
                      id='download_history_data_task_14_57')
    # 然后在14:58执行buying_strategy_task，确保数据下载完成  
    scheduler.add_job(buying_strategy_task, 'cron', day_of_week='mon-fri', hour='14', minute='58',
                      id='buying_strategy_task')

    # 4. 每个交易日9:20、11:35、15:05执行 generate_trading_report。  
    report_times = ['9:20', '11:35', '15:05']
    for rt in report_times:
        hour, minute = map(int, rt.split(':'))
        scheduler.add_job(generate_trading_report_task, 'cron', day_of_week='mon-fri', hour=hour, minute=minute,
                          id=f'generate_trading_report_{rt}')

        # 5. 每个交易日交易时间段内，每10分钟检查一次 stop_loss_main 是否运行，若未运行则启动。
    # 注意交易时间段为 9:30-11:30 和 13:00-15:00  
    # 设置在 9:30 到 11:30 和 13:00 到 15:00，每隔10分钟执行一次  
    scheduler.add_job(stop_loss_main_task, 'cron', day_of_week='mon-fri', hour='9-11', minute='30-59/10',
                      id='stop_loss_main_task_morning')
    scheduler.add_job(stop_loss_main_task, 'cron', day_of_week='mon-fri', hour='13-14', minute='0-59/10',
                      id='stop_loss_main_task_afternoon')

    # 6. 每个交易日8:30-16:00，每10分钟执行一次 start_xt_client。  
    scheduler.add_job(start_xt_client_task, 'cron', day_of_week='mon-fri', hour='8-15', minute='*/10',
                      id='start_xt_client_task')

    # 启动调度器  
    scheduler.start()
    logger.info("任务调度器已启动。按 Ctrl+C 退出。")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("任务调度器已关闭。")
