

def test_get_targets():
    from utils.data import get_targets_list_from_csv
    target_list = get_targets_list_from_csv()
    print("*" * 10)
    print(target_list)
    print(len(target_list))


def test_get_stock_data():
    from load_data.download_xt_data import get_data_from_local
    df = get_data_from_local()
    print('\n')
    print(df.head())
    print(df.columns)
    print(df.shape)


def test_get_data_from_local():
    from load_data.download_xt_data import get_data_from_local
    data = get_data_from_local()
    data.to_csv("data/stock_data.csv")


def test_generate_processed_series_data():
    from load_data.multivariate_timeseries import generate_processed_series_data
    generate_processed_series_data()


def test_read_max_profile():
    from strategy.stop_loss import StopLossProgram
    p = StopLossProgram()
    p.load_max_profit()
    print(p.max_profit)

def test_start_xttrader():
    from strategy.trader import setup_xt_trader
    xt_trader = setup_xt_trader()

