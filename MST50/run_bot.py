# run_bot.py

"""
This module contains the main function to execute strategies on every minute.
The on_minute function runs every minute and checks if a new hour has started, in backtesting mode: will run per the backtest step and simulation time.
Also per strategy, it checks if a new bar has started and executes the strategy accordingly.
Functions:
    on_minute: Main function to execute strategies on every minute.

"""
import os

from .utils import (
    write_balance_performance_file, is_new_bar, wait_for_new_minute,
    print_hashtaged_msg, print_with_info
)
from .symbols import Timeframe

# Always import account_info, shutdown, and last_error from mt5_interface
from .mt5_interface import account_info, shutdown, last_error, time_current

def on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict, BACKTEST_MODE):
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
    now = time_current()
    if not BACKTEST_MODE:
        print(f"on_minute strarted, ", f"Current time: {now}")

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
            #TODO: check if this is correct
            open_trades = [trade for strategy in strategies.values() for trade in strategy.open_trades]
            write_balance_performance_file(account_info_dict, open_trades)
        # Failed to get account info
        else:
            print_hashtaged_msg(3, "Failed to get account info", "Failed to get account info, error code =", last_error())

    def execute_strategy(strategy, symbols, time_bar, new_hour, account_info_dict):
        if is_new_bar(strategy.timeframe, time_bar) or new_hour: # Check if a new bar has started, or if a new hour has started - new hour because for daily strategies, we can have the logic run on a certain hour - and not 00:00
            #print(f"New bar detected for strategy:{strategy.strategy_num}-{strategy.strategy_name} strategy timeframe: {strategy.str_timeframe}")
            strategy.handle_new_bar(symbols)
            strategy.write_strategy_performance_file(account_info_dict)
        else:
            strategy.handle_new_minute(symbols)

            

    [execute_strategy(strategy, symbols, time_bar, new_hour, account_info_dict) for strategy in strategies.values()]

    if not BACKTEST_MODE and time_bar.check_last_minute_of_hour(): # Last minute of the hour - only relvelant for live trading
        print_hashtaged_msg(5, "on_minute", "Last minute of the hour, waiting for new hour to start...")
        # Rebalance once an hour - Make sure that each run of on_minute is at the start of a new minute
        wait_for_new_minute(time_bar)
        # Run once immediately (at the start of the new hour)
        on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict, BACKTEST_MODE)
