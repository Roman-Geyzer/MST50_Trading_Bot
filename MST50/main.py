# main.py

"""
This script is the main entry point for executing trading strategies using the provided configuration.
Functions:
    main(): Main function to execute trading strategies in either live trading or backtesting mode.
Modules:
    schedule: Python job scheduling for running the on_minute function every minute.
    time: Time access and conversions.
    mt5_interface: Interface to MetaTrader 5 for trading and data access.
    strategy: Module for defining trading strategies.
    symbols: Module for defining trading symbols.
    Backtest.main_backtest: Module for running backtesting.
    utils: Utility functions for logging and printing messages.
    run_bot: Module for running the on_minute function.
Constants:
    run_mode (list): Specifies the modes in which the trading strategies can run, either 'live' or 'demo'.
    cores (int): Number of cores to use for parallel processing.
    pytest_count (int): Number of times the pytest module has been run.
"""

import os

# Set BACKTEST_MODE to 'True' for backtesting, 'False' for live trading
os.environ['BACKTEST_MODE'] = 'True'  # Change to 'True' when ready to backtest

# Determine if we are in backtesting mode
BACKTEST_MODE = os.environ.get('BACKTEST_MODE', 'False') == 'True'

import schedule
import time

from .strategy import Strategy
from .symbols import Symbol
from .Backtest.main_backtest import main_backtest

# no need for conditional import - always import the following modules (backtest will import the other modules)
from .utils import TradeHour, TimeBar, print_hashtaged_msg, print_with_info, wait_for_new_minute

# Always import account_info, shutdown, and last_error from mt5_interface
from .mt5_interface import account_info, shutdown, last_error
from .run_bot import on_minute

run_mode = ['dev']



#TODO: create full seperation of the main function for live trading and backtesting
def main():
    """
    Main function to execute trading strategies.
    will run the main_backtest function if BACKTEST_MODE is True, otherwise will run the live trading mode.
    """
    print_hashtaged_msg(1, "Initializing MST50", "Initializing MST50...")
    # Initialize strategies and symbols
    strategies = Strategy.initialize_strategies(run_mode)
    if BACKTEST_MODE:
        main_backtest(strategies)

    # Live trading mode (no need for else since main backtes has quit the program)
    symbols = Symbol.initialize_symbols(strategies)
    account_info_dict = account_info()
    print("Account info:", account_info_dict)

    # Initialize the previous hour and day to -1 to ensure the first iteration runs the on_hour function
    trade_hour = TradeHour()
    time_bar = TimeBar()

    print(f"Initialized trade hour, hour= {trade_hour.current_hour}, day= {trade_hour.current_day}")
    print(f"Initialized time bar, current_bar (highest new bar) = {time_bar.current_bar}")
    # Schedule the on_minute function to run every minute
    print_hashtaged_msg(1, "Scheduling on_minute", "Scheduling on_minute function to run every minute...")

    ############################################
    #    TODO: uncomment the following line    #
    ############################################
    wait_for_new_minute(time_bar)  # Make sure that each run of on_minute is at the start of a new minute
    on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict)  # Run once immediately
    schedule.every(1).minutes.do(
        on_minute,
        strategies=strategies,
        trade_hour=trade_hour,
        time_bar=time_bar,
        symbols=symbols,
        account_info_dict=account_info_dict
    )

    print_hashtaged_msg(1, "Initialization Complete", "All initializations completed successfully, waiting for new minute to start executing strategies...")
    # Run: execute strategies
    try:
        while True:
            schedule.run_pending()  # Check if any scheduled tasks need to run
            time.sleep(1)           # Sleep for 1 second before checking again
    except KeyboardInterrupt:
        print_hashtaged_msg(3, "Keyboard Interrupt", "Stopping strategies...")

    finally:
        # Shutdown MetaTrader 5 connection
        shutdown()

if __name__ == "__main__":
    main()