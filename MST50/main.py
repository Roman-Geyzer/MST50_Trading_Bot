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
    run_modes (list): Specifies the modes in which the trading strategies can run, either 'live' or 'demo'.
    cores (int): Number of cores to use for parallel processing.
    pytest_count (int): Number of times the pytest module has been run.
"""

import os




import time

from .strategy import Strategy
from .symbols import Symbol
from .Backtest.main_backtest import main_backtest

# no need for conditional import - always import the following modules (backtest will import the other modules)
from .utils import TradeHour, TimeBar, print_hashtaged_msg, initialize_balance_performance_file, wait_for_new_minute

# Always import account_info, shutdown, and last_error from mt5_interface
from .mt5_interface import account_info, shutdown, last_error
from .run_bot import on_minute





#TODO: create full seperation of the main function for live trading and backtesting
def main(run_modes: list = ['live'], BACKTEST_MODE: bool = False):
    """
    Main function to execute trading strategies.
    will run the main_backtest function if BACKTEST_MODE is True, otherwise will run the live trading mode.
    """
    print("Initializing MST50", "Initializing MST50...")
    # Initialize strategies and symbols
    strategies = Strategy.initialize_strategies(run_modes)
    if BACKTEST_MODE:
        main_backtest(strategies)

    # Live trading mode (no need for else since main backtes has quit the program)

    # Initialize required indicators based on strategies

    from .mt5_client import initialize_required_indicator_columns
    initialize_required_indicator_columns(strategies)
    
    symbols = Symbol.initialize_symbols(strategies)
    account_info_dict = account_info()
    print("Account info:", account_info_dict)

    initialize_balance_performance_file()

    # Initialize the previous hour and day to -1 to ensure the first iteration runs the on_hour function
    trade_hour = TradeHour()
    time_bar = TimeBar()

    print(f"Initialized trade hour, hour= {trade_hour.current_hour}, day= {trade_hour.current_day}")
    print(f"Initialized time bar, current_bar (highest new bar) = {time_bar.current_bar}")
    print(f"run on_minute once immediately after initialization")
    on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict,BACKTEST_MODE)  # Run once immediately after initialization 

    print_hashtaged_msg(1, "Initialization Complete", "All initializations completed successfully, waiting for new minute to start executing strategies...")

    # TODO: change logic to send email (or other method) when an error occurs, also add one hour wait and try again (decide on the number of retries)
    try:
        while True:     # Schedule the on_minute function to run every minute
            wait_for_new_minute(time_bar)  # Make sure that each run of on_minute is at the start of a new minute
            on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict,BACKTEST_MODE)  # Run 
            time.sleep(45) # sleep for 45 seconds - no need to run the wait_for_new_minute which uses time.sleep(1)
    except KeyboardInterrupt:
        print_hashtaged_msg(3, "Keyboard Interrupt", "Stopping strategies...")

    finally:
        # Shutdown MetaTrader 5 connection
        shutdown()

if __name__ == "__main__":
    main()