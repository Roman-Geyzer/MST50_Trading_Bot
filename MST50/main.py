# main.py

"""
main.py
This script is the main entry point for executing trading strategies using the provided configuration.
Functions:
    main(): Initializes MetaTrader 5, loads strategy configurations, schedules strategy execution, and manages the execution loop.
Modules:
    strategy: Contains the Strategy class used for executing trading strategies.
    schedule: Used for scheduling tasks at specific intervals.
    time: Provides time-related functions.
    datetime: Supplies classes for manipulating dates and times.
    pandas: Provides data structures for working with data.
    pprint: Provides pretty-printing of data structures.
    concurrent: Provides support for asynchronous execution.
    sys: Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
    os: Provides a way of using operating system dependent functionality.
    constants: Contains constants used throughout the project.
    symbols: Contains the Symbol class used for storing symbol data.
    utils: Contains utility functions used throughout the project.
    Pyro5: Provides a way to create remote objects in Python - used for initializing the TradingPlatform class. (first use MT5)
    ThreadPoolExecutor: Provides a high-level interface for asynchronously executing functions in separate threads.

Constants:
    run_mode (list): Specifies the modes in which the trading strategies can run, either 'live' or 'demo'.
"""



"""
This script is the main entry point for executing trading strategies using the provided configuration.
Functions:
    main(): Initializes MetaTrader 5, loads strategy configurations, schedules strategy execution, and manages the execution loop.
    on_minute(): Executes trading strategies on every minute.
    execute_strategy(): Executes a single strategy based on the current minute.
Modules:   
    strategy: Contains the Strategy class used for executing trading strategies.
    schedule: Used for scheduling tasks at specific intervals.
    time: Provides time-related functions.
    datetime: Supplies classes for manipulating dates and times.
    pandas: Provides data structures for working with data.
    pprint: Provides pretty-printing of data structures.
    concurrent: Provides support for asynchronous execution.
    sys: Provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
    os: Provides a way of using operating system dependent functionality.
    constants: Contains constants used throughout the project.
    symbols: Contains the Symbol class used for storing symbol data.
    utils: Contains utility functions used throughout the project.
    Pyro5: Provides a way to create remote objects in Python - used for initializing the TradingPlatform class.
Constants:
    run_mode (list): Specifies the modes in which the trading strategies can run, either 'live' or 'demo'.
    cores (int): Number of cores to use for parallel processing.
    strategy_timeout (int): Time limit in seconds for executing a strategy.
    pytest_count (int): Number of times the pytest module has been run.
"""


import schedule
import time
import pandas as pd


from .strategy import Strategy
from .utils import (write_balance_performance_file, TradeHour, is_new_bar, TimeBar,
                    wait_for_new_minute, print_hashtaged_msg)
from .symbols import Symbol, Timeframe

from .mt5_client import account_info, shutdown, last_error




run_mode = ['dev']
cores = 3
# TOSO: change the strategy timeout to 20 seconds
strategy_timeout = 950

pytest_count = 0


                

def on_minute(strategies, trade_hour,time_bar, symbols, account_info_dict):
    """
    main function to execute strategies on every minute
    funtion runs on every minute and checks if a new hour has started
    also per strategy, it checks if a new bar has started and executes the strategy accordingly
    Args:
        strategies (dict): Dictionary containing strategy instances.
        trade_hour (TradeHour): TradeHour instance to track the current hour and day.
        time_bar (TimeBar): TimeBar instance to track the current bar timeframe.
        symbols (dict): Dictionary containing symbol instances with their respective timeframes and rates.
    """
    print_hashtaged_msg(5, "on_minute", "on_minute function started...")
    # Fetch rates for all symbols and timeframes - the metho will only update the rates if a new bar has started
    time_bar.update_tf_bar()
    Timeframe.fetch_new_bar_rates(symbols ,time_bar) # Fetch new bar rates for all symbols and all *new* timeframes
    account_info_dict = account_info()


    # Check if a new hour has started - if so, start new hour logic
    new_hour = trade_hour.is_new_hour()
    if new_hour:
        print(f"New hour: {trade_hour.current_hour}, day: {trade_hour.current_day}")
        account_info_dict = account_info()
        
        if account_info is not None:
            write_balance_performance_file(account_info_dict)
        # failed to get account info
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
        wait_for_new_minute(time_bar) # Rebalance once an hour - Make sure that each run of on_minute is at the start of a new minute
        on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict) # Run once immediately (at the start of the new hour)




def main():
    """
    main function to execute trading strategies
    
    """
    print_hashtaged_msg(1, "Initializing MST50", "Initializing MST50...")
    # Initialize MetaTrader 5 connection, strategies, and symbols
    #TODO: redo the initialization of the MT5 connection using Pyro5
    #initialize_mt5()
    strategies = Strategy.initialize_strategies(run_mode)
    symbols = Symbol.initialize_symbols(strategies)

    # Initialize the previous hour and day to -1 to ensure the first iteration runs the on_hour function
    trade_hour = TradeHour()
    time_bar = TimeBar()

    print(f"Initialized trade hour, hour= {trade_hour.current_hour}, day= {trade_hour.current_day}")
    print(f"Initialized time bar, curret_bar (highest new bar) = {time_bar.current_bar}")
    account_info_dict = account_info()
    print("Account info:", account_info_dict)

    # Schedule the on_minute function to run every minute
    print_hashtaged_msg(1, "Scheduling on_minute", "Scheduling on_minute function to run every minute...")

    ############################################
    #    TODO: uncomment the following line    #
    ############################################
   # wait_for_new_minute(time_bar) # Make sure that each run of on_minute is at the start of a new minute
    on_minute(strategies, trade_hour,time_bar, symbols, account_info_dict) # Run once immediately
    schedule.every(1).minutes.do(on_minute, strategies=strategies, trade_hour=trade_hour, time_bar=time_bar, symbols=symbols, account_info_dict=account_info_dict)

    print_hashtaged_msg(1, "Initialization Complete", "All initializations completed successfully., waiting for new minute to start executing strategies...")
    # run: execute strategies
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

