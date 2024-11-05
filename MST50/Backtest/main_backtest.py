# Backtest/main_backtest.py
"""
This module contains the main backtesting function that runs the backtesting loop.
The backtesting loop advances the simulation time and executes strategies.
functions:
    run_backtest_loop: Run the backtesting loop, advancing the simulation time and executing strategies.
    main_backtest: Main backtest function that initializes the backtest and runs the backtesting loop.
"""
from .mt5_backtest import MT5Backtest, initialize_backtest, shutdown
from ..run_bot import on_minute
from .time_backtest import TradeHour, TimeBar
from ..symbols import Symbol
from ..mt5_interface import account_info

def run_backtest_loop(strategies, trade_hour, time_bar, symbols, backtest):
    """
    Run the backtesting loop, advancing the simulation time and executing strategies.
    Args:
        strategies (list): List of strategies to execute.
        trade_hour (TradeHour): TradeHour object to track the current trade hour.
        time_bar (TimeBar): TimeBar object to track the current time bar.
        symbols (list): List of symbols to trade.
        backtest (MT5Backtest): MT5Backtest object to run the backtest.
    """
    while backtest.current_time < backtest.end_time:
        # Advance the simulation time
        proceed = backtest.step_simulation()
        if not proceed:
            print("Backtest completed.")
            break
         # Call the on_new_bar function to process strategies
        on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict=None)
    # Finalize the backtest
    backtest.export_logs()
    # Shutdown Batckest - clear memory ext.
    shutdown()


def main_backtest(strategies):
        """
        Main backtest function that initializes the backtest and runs the backtesting loop.
        Args:
            strategies (list): List of strategies to execute.
        """
        initialize_backtest(strategies)
        backtest = MT5Backtest(strategies=strategies)
        symbols = Symbol.initialize_symbols(strategies)
        account_info_dict = account_info()
        print("Account info:", account_info_dict)
        # Initialize TradeHour and TimeBar with backtest
        trade_hour = TradeHour(backtest)
        time_bar = TimeBar(backtest)
        # Run backtesting loop
        run_backtest_loop(strategies, trade_hour, time_bar, symbols, backtest)