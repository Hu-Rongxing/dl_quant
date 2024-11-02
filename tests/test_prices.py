def test_get_max_price():
    from utils.data import get_max_ask_price
    # 不开交易的市场指数
    get_max_ask_price("000001.SZ")
    # 可以交易的股票价格
    get_max_ask_price("159998.SZ")
    # 停牌
    get_max_ask_price("603305.SH")
