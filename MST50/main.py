# main.py

"""
This script is the main entry point for executing trading strategies using the provided configuration.
Functions:
    main(): Initializes MetaTrader 5, loads strategy configurations, schedules strategy execution, and manages the execution loop.
    on_minute(): Executes trading strategies on every minute.
    run_backtest_loop(): Runs the backtesting loop, advancing the simulation time and executing strategies.
Modules:
    strategy: Contains the Strategy class used for executing trading strategies.
    schedule: Used for scheduling tasks at specific intervals.
    time: Provides time-related functions.
    datetime: Supplies classes for manipulating dates and times.
    os: Provides a way of using operating system dependent functionality.
    symbols: Contains the Symbol class used for storing symbol data.
    utils: Contains utility functions used throughout the project.
Constants:
    run_mode (list): Specifies the modes in which the trading strategies can run, either 'live' or 'demo'.
    cores (int): Number of cores to use for parallel processing.
    strategy_timeout (int): Time limit in seconds for executing a strategy.
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

# Conditional import of TradeHour and TimeBar based on the mode
if BACKTEST_MODE:
    from .time_backtest import TradeHour, TimeBar
    from .mt5_backtest import MT5Backtest, initialize_backtest
else:
    from .utils import TradeHour, TimeBar

from .utils import (
    write_balance_performance_file, is_new_bar, wait_for_new_minute,
    print_hashtaged_msg, print_with_info
)
from .symbols import Symbol, Timeframe

# Always import account_info, shutdown, and last_error from mt5_interface
from .mt5_interface import account_info, shutdown, last_error

run_mode = ['dev']
cores = 3
# TODO: change the strategy timeout to 20 seconds
strategy_timeout = 950
pytest_count = 0

def on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict):
    """
    Main function to execute strategies on every minute.
    Function runs every minute and checks if a new hour has started.
    Also per strategy, it checks if a new bar has started and executes the strategy accordingly.
    Args:
        strategies (dict): Dictionary containing strategy instances.
        trade_hour (TradeHour): TradeHour instance to track the current hour and day.
        time_bar (TimeBar): TimeBar instance to track the current bar timeframe.
        symbols (dict): Dictionary containing symbol instances with their respective timeframes and rates.
        account_info_dict (dict): Account information dictionary.
    """
    print_hashtaged_msg(1, "on_minute", "on_minute function started...")
    # Fetch rates for all symbols and timeframes - the method will only update the rates if a new bar has started
    time_bar.update_tf_bar()
    Timeframe.fetch_new_bar_rates(symbols, time_bar)  # Fetch new bar rates for all symbols and all *new* timeframes
    account_info_dict = account_info()

    # Check if a new hour has started - if so, start new hour logic
    new_hour = trade_hour.is_new_hour()
    if new_hour:
        print(f"New hour: {trade_hour.current_hour}, day: {trade_hour.current_day}")
        account_info_dict = account_info()

        if account_info_dict is not None:
            write_balance_performance_file(account_info_dict)
        # Failed to get account info
        else:
            print_hashtaged_msg(3, "Failed to get account info", "Failed to get account info, error code =", last_error())

    def execute_strategy(strategy, symbols, time_bar, new_hour, account_info_dict):
        if is_new_bar(strategy.timeframe, time_bar):
            print(f"New bar detected for strategy:{strategy.strategy_num}-{strategy.strategy_name} strategy timeframe: {strategy.str_timeframe}")
            strategy.handle_new_bar(symbols)
        else:
            strategy.handle_new_minute(symbols)
        if new_hour:
            strategy.write_strategy_performance_file(account_info_dict)

    [execute_strategy(strategy, symbols, time_bar, new_hour, account_info_dict) for strategy in strategies.values()]

    if time_bar.check_last_minute_of_hour():
        print_hashtaged_msg(5, "on_minute", "Last minute of the hour, waiting for new hour to start...")
        # Rebalance once an hour - Make sure that each run of on_minute is at the start of a new minute
        wait_for_new_minute(time_bar)
        # Run once immediately (at the start of the new hour)
        on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict)



def on_new_bar(strategies, trade_hour, time_bar, symbols, account_info_dict):
    """
    Main function to backtest strategies on every mnew bar (per backtest settings)
    Function runs on simulated new bars
    Also per strategy, it checks if a new bar has started and executes the strategy accordingly.
    Args:
        strategies (dict): Dictionary containing strategy instances.
        trade_hour (TradeHour): TradeHour instance to track the current hour and day.
        time_bar (TimeBar): TimeBar instance to track the current bar timeframe.
        symbols (dict): Dictionary containing symbol instances with their respective timeframes and rates.
        account_info_dict (dict): Account information dictionary.
    """
    time_bar.update_tf_bar()
    Timeframe.fetch_new_bar_rates(symbols, time_bar)  # Fetch new bar rates for all symbols and all *new* timeframes
    account_info_dict = account_info()

    # Check if a new hour has started - if so, start new hour logic
    new_hour = trade_hour.is_new_hour()
    if new_hour:
        write_balance_performance_file(account_info_dict)

    def execute_strategy(strategy, symbols, time_bar, new_hour, account_info_dict):
        if is_new_bar(strategy.timeframe, time_bar):
            strategy.handle_new_bar(symbols)
        else:
            strategy.handle_new_minute(symbols)
        if new_hour:
            strategy.write_strategy_performance_file(account_info_dict)

    [execute_strategy(strategy, symbols, time_bar, new_hour, account_info_dict) for strategy in strategies.values()]

def run_backtest_loop(strategies, trade_hour, time_bar, symbols, backtest):
    """
    Run the backtesting loop, advancing the simulation time and executing strategies.
    """
    try:
        while backtest.current_time < backtest.end_time:
            # Advance the simulation time
            proceed = backtest.step_simulation()
            if not proceed:
                print("Backtest completed.")
                break

            # The TradeHour and TimeBar classes automatically update current_time from backtest.current_time

            # Call the on_new_bar function to process strategies
            on_new_bar(strategies, trade_hour, time_bar, symbols, account_info_dict=None)
    except Exception as e:
        print_hashtaged_msg(3, "Backtest Error", f"An error occurred during backtesting: {e}")
    finally:
        # Finalize the backtest
        backtest.export_logs()
        # Shutdown MetaTrader 5 connection
        shutdown()

#TODO: create full seperation of the main function for live trading and backtesting
def main():
    """
    Main function to execute trading strategies.
    """
    print_hashtaged_msg(1, "Initializing MST50", "Initializing MST50...")
    # Initialize strategies and symbols
    strategies = Strategy.initialize_strategies(run_mode)
    if not BACKTEST_MODE:
        symbols = Symbol.initialize_symbols(strategies)
        account_info_dict = account_info()
        print("Account info:", account_info_dict)

    if BACKTEST_MODE:
        # Initialize backtest with strategies
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
    else: # Live trading mode
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
        # wait_for_new_minute(time_bar)  # Make sure that each run of on_minute is at the start of a new minute
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