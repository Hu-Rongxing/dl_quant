

def test_logger():
    from utils.logger import logger, log_manager

    try:
        logger.debug("这是一条调试信息")
        logger.info("这是一条信息")
        logger.warning("这是一条警告信息")

        for _ in range(5):
            logger.trader("这是一条 TRADER 级别的信息")
        logger.error("这是一条错误信息")

        1 / 0
    except ZeroDivisionError:
        logger.exception("捕获到异常")
    finally:
        log_manager.stop()