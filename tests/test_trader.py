from turtledemo.penrose import start

from monitor import buying_strategy_task

def test_buying_strategy_task():
    buying_strategy_task()


def test_restart_client():
    from strategy.qmt_monitor import start_xt_client, ProgramMonitor
    start_xt_client()
    xt_client = ProgramMonitor()
    xt_client.restart_program()
