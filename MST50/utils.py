# utils.py
"""
This module contains utility functions for the MST50 package.
Classes:
    TradeHour: Class to track the current hour and day.
    TimeBar: Class to track the current bar for different timeframes.
Methods:
    TradeHour:
        is_new_hour: Check if a new hour has started.
        is_new_day: Check if a new day has started.
        is_new_week: Check if a new week has started.
        update_current_day: Update the current day.
        update_current_hour: Update the current hour.
    TimeBar:
        update_tf_bar: Check the highest timeframe that has a new bar and update the respective attributes.
        check_last_minute_of_hour: Check if the last minute of the hour has been reached.

Variables:
    magic_number_base: Base magic number for strategies.
    performance_file: File to write performance data.
    TIMEFRAME_MAGIC_NUMBER_MAPPING: Mapping of timeframes to magic numbers.
    SYMBOL_MAGIC_NUMBER_MAPPING: Mapping of symbols to magic numbers.
    TIMEFRAME_MT5_MAPPING: Mapping of timeframes to MetaTrader 5 timeframes.
    TIMEFRAME_STRING_MAPPING: Mapping of timeframes to string representations.
    BarsTFs: Enum for bar timeframes.
Functions:
    load_config: Load configuration from an Excel file into a structured dictionary - used for strategies configuration and initialization.
    get_mt5_timeframe: Get the MetaTrader 5 timeframe for a given timeframe.
    get_timeframe_string: Get the string representation of a timeframe.
    get_timeframe_magic_number: Get the magic number for a timeframe.
    get_final_magic_number: Get the final magic number for a strategy.
    str_to_bool: Convert a value to boolean.
    safe_int_convert: Safely convert a value to an integer.
    safe_float_convert: Safely convert a value to a float.
    safe_date_convert: Safely convert a value to a datetime.
    safe_int_extract_from_dict: Safely extract an integer value from a dictionary.
    safe_float_extract_from_dict: Safely extract a float value from a dictionary.
    safe_bool_extract_from_dict: Safely extract a boolean value from a dictionary.
    safe_str_extract_from_dict: Safely extract a string value from a dictionary.
    write_balance_performance_file: Write the balance and performance data to a CSV file.
    wait_for_new_minute: Wait for a new minute to start based on the TimeBar object.
    print_with_info: Prints information about the call stack up to 'levels_up' levels.
    attempt_with_stages_and_delay: Attempt to execute a function a specified number of times with a delay between attempts.
    catch_i_times_with_s_seconds_delay: Attempt to execute a function a specified number of times with a delay between attempts.
    calculate_history_length: Calculate the history length based on the strategy configuration.
Functions:

"""

import pandas as pd
import inspect
from datetime import datetime


from .constants import (magic_number_base, TIMEFRAME_MAGIC_NUMBER_MAPPING,
                       SYMBOL_MAGIC_NUMBER_MAPPING, TIMEFRAME_MT5_MAPPING, TIMEFRAME_STRING_MAPPING)
import time
from .mt5_interface import TIMEFRAMES, copy_rates_from, time_current
import math
import platform
import os

#TODO: update the perfomance file to include broker and account number

# Determine if we are in backtesting mode
BACKTEST_MODE = os.environ.get('BACKTEST_MODE', 'False') == 'True'

if platform.system() == 'Windows':
    drive = "x:\\"
else:
    drive = "/Volumes/TM"

# check if drive is accesale - if not then assign drive to c:
if not os.path.exists(drive):
    print(f"Drive {drive} is not accessible. Reassigning to 'c:'.")
    drive = "c:\\"  # Reassign to 'c:' if the drive is not accessible

performance_dir = os.path.join(drive, 'documentation', 'account')


performance_file = os.path.join(performance_dir, 'performance.csv')



class TradeHour:
    def __init__(self):
        self.current_hour = -1
        self.current_day = -1

    def is_new_hour(self):
        current_time = time_current()
        if self.current_hour != current_time.hour or self.current_day != current_time.day:
            self.current_hour = current_time.hour
            self.current_day = current_time.day
            return True
        return False
    def is_new_day(self):
        current_time = time_current()
        if self.current_day != current_time.day:
            self.current_day = current_time.day
            return True
        return False
    
    def is_new_week(self):
        current_time = time_current()
        if self.current_day != current_time.isocalendar()[1]:
            self.current_day = current_time.isocalendar()[1]
            return True
        return False
    
    def update_current_day(self):
        self.current_day = time_current().day

    def update_current_hour(self):
        self.current_hour = time_current().hour

class TimeBar:
    def __init__(self):
        self.M1 = -1
        self.M5 = -1
        self.M15 = -1
        self.M30 = -1
        self.H1 = -1
        self.H4 = -1
        self.D1 = -1
        self.W1 = -1
        self.current_bar = self.update_tf_bar()
    
    def update_tf_bar(self):
        """
        Checks the highest timeframe that has a new bar and updates the respective attributes.
        Returns the timeframe that had the latest update.
        """
        current_time = time_current()  # Get the current time
        current_week = current_time.isocalendar()[1]  # ISO calendar week number
        # Check for a new weekly bar
        if self.W1 != current_week:
            self.W1 = current_week
            self.D1 = current_time.day
            self.H4 = current_time.hour // 4
            self.H1 = current_time.hour
            self.M30 = current_time.minute // 30
            self.M15 = current_time.minute // 15
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "W1"
            return "W1"

        # Check for a new daily bar
        if self.D1 != current_time.day:
            self.D1 = current_time.day
            self.H4 = current_time.hour // 4
            self.H1 = current_time.hour
            self.M30 = current_time.minute // 30
            self.M15 = current_time.minute // 15
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "D1"
            return "D1"

        # Check for a new 4-hour bar
        if self.H4 != current_time.hour // 4:
            self.H4 = current_time.hour // 4
            self.H1 = current_time.hour
            self.M30 = current_time.minute // 30
            self.M15 = current_time.minute // 15
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "H4"
            return "H4"

        # Check for a new hourly bar
        if self.H1 != current_time.hour:
            self.H1 = current_time.hour
            self.M30 = current_time.minute // 30
            self.M15 = current_time.minute // 15
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "H1"
            return "H1"

        # Check for a new 30-minute bar
        if self.M30 != current_time.minute // 30:
            self.M30 = current_time.minute // 30
            self.M15 = current_time.minute // 15
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "M30"
            return "M30"

        # Check for a new 15-minute bar
        if self.M15 != current_time.minute // 15:
            self.M15 = current_time.minute // 15
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "M15"
            return "M15"

        # Check for a new 5-minute bar
        if self.M5 != current_time.minute // 5:
            self.M5 = current_time.minute // 5
            self.M1 = current_time.minute
            self.current_bar = "M5"
            return "M5"

        # Check for a new 1-minute bar
        if self.M1 != current_time.minute:
            self.M1 = current_time.minute
            self.current_bar = "M1"
            return "M1"

        # If no timeframe has changed, return None
        return None

    def check_last_minute_of_hour(self):
        print("check_last_minute_of_hour")
        print(f"self.M1 = {self.M1}")
        return self.M1 == 59





def is_new_bar(timeframe, time_bar):
    """
    Check if a new bar has formed for the given timeframe.
    Args:
        timeframe (str): The timeframe to check for a new bar.
    Returns:
        bool: True if a new bar has formed, False otherwise.
    """
    timeframe = get_timeframe_string(timeframe)
    timeframe_list = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
    if timeframe_list.index(timeframe) <= timeframe_list.index(time_bar.current_bar): # check if the timeframe is lower or equal to the current bar - if so, new bar
        return True
    return False


def get_mt5_timeframe(timeframe):
    # Return the corresponding MetaTrader 5 timeframe for the given timeframe
    return TIMEFRAME_MT5_MAPPING.get(timeframe, None)

def get_timeframe_string(timeframe):
    # Return the corresponding index for the given timeframe
    return TIMEFRAME_STRING_MAPPING.get(timeframe, None)

def get_timeframe_magic_number(timeframe):
    # Return the corresponding number for the given timeframe, will retrun 0 by default = error (need to check)
    time_frame_magic_number =  TIMEFRAME_MAGIC_NUMBER_MAPPING.get(timeframe, 0)
    if time_frame_magic_number == 0:
        print(f"Timeframe magic number not found for timeframe: {timeframe}")
        raise ValueError(f"Timeframe magic number not found for timeframe: {timeframe}")
    return time_frame_magic_number


def get_final_magic_number(symbol,strategy_magic_number):
    symbol_magic_number = SYMBOL_MAGIC_NUMBER_MAPPING.get(symbol, 0)
    if symbol_magic_number == 0:
        print(f"Symbol magic number not found for symbol: {symbol}")
        return None
    return strategy_magic_number + symbol_magic_number

def str_to_bool(s):
    """
    Convert a value to boolean.
    Handles both boolean and string representations of truthy/falsy values.
    
    Parameters:
        s (str or bool): The value to convert.

    Returns:
        bool: The converted boolean value.
    """
    if isinstance(s, bool):
        return s
    return str(s).lower() == 'true'


def safe_int_convert(value, default=0):
    """
    Safely convert a value to an integer.
    If conversion fails, return a default value.

    Parameters:
        value: The value to convert.
        default (int): The default value to return if conversion fails.

    Returns:
        int: The converted integer value or the default value.
    """
    try:
        float_value = float(value) # Convert the string to a float
        int_value = int(float_value)# Convert the float to an integer
        return int_value
    except (ValueError, TypeError):
        return default


def safe_float_convert(value, default=0.0):
    """
    Safely convert a value to a float.
    If conversion fails, return a default value.

    Parameters:
        value: The value to convert.
        default (float): The default value to return if conversion fails.

    Returns:
        float: The converted float value or the default value.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
    


def safe_date_convert(value):
    """
    Safely convert a value to a datetime.
    If conversion fails, raise an error.

    Parameters:
        value: The value to convert.

    Returns:
        datetime: The converted datetime value.
    """
    try:
        return pd.to_datetime(value).to_pydatetime()
    except (ValueError, TypeError) as e:
        print(f"Error converting value to datetime: {e}")
        raise

def safe_int_extract_from_dict(dict, key, default=0):
    """
    Safely extract an integer value from a dictionary.
    If the key is not present or the value cannot be converted to an integer, return a default value.

    Parameters:
        dict (dict): The dictionary to extract the value from.
        key: The key to extract the value for.
        default (int): The default value to return if extraction fails.

    Returns:
        int: The extracted integer value or the default value.
    """
    try:
        return int(dict[key])
    except (KeyError, ValueError, TypeError):
        return default

def safe_float_extract_from_dict(dict, key, default=0.0):
    """
    Safely extract a float value from a dictionary.
    If the key is not present or the value cannot be converted to a float, return a default value.

    Parameters:
        dict (dict): The dictionary to extract the value from.
        key: The key to extract the value for.
        default (float): The default value to return if extraction fails.

    Returns:
        float: The extracted float value or the default value.
    """
    try:
        return float(dict[key])
    except (KeyError, ValueError, TypeError):
        return default

def safe_bool_extract_from_dict(dict, key, default=False):
    """
    Safely extract a boolean value from a dictionary.
    If the key is not present or the value cannot be converted to a boolean, return a default value.

    Parameters:
        dict (dict): The dictionary to extract the value from.
        key: The key to extract the value for.
        default (bool): The default value to return if extraction fails.

    Returns:
        bool: The extracted boolean value or the default value.
    """
    try:
        return str_to_bool(dict[key])
    except (KeyError, ValueError, TypeError):
        return default

def safe_str_extract_from_dict(dict, key, default=''):
    """
    Safely extract a string value from a dictionary.
    If the key is not present or the value cannot be converted to a string, return a default value.

    Parameters:
        dict (dict): The dictionary to extract the value from.
        key: The key to extract the value for.
        default (str): The default value to return if extraction fails.

    Returns:
        str: The extracted string value or the default value.
    """
    try:
        return str(dict[key])
    except (KeyError, ValueError, TypeError):
        return default


def load_config(sheet_name='config', strategies_run_modes=['live']):
    """
    Load configuration from an Excel file into a structured dictionary.
    This loader reads the specified sheet and processes each row into a dictionary.

    Parameters:
        filename (str): The name of the Excel file to read.
        sheet_name (str): The name of the sheet in the Excel file to read.
        strategies_run_modes (list): List of strategy statuses to load (e.g., ['live', 'demo']).

    Returns:
        dict: A dictionary containing strategy configurations.
    """
    # Construct the full path to the configuration file
    
    
    # Read the specified sheet into a DataFrame and handle missing values
    df = pd.read_excel("config.xlsx", sheet_name="config")
    import numpy as np
    df.replace({np.nan: None}, inplace=True)


    # Initialize an empty dictionary to hold strategy configurations
    strategies_config = {}

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():

        # Filter strategies based on the run mode
        if row['strategy_status'] not in strategies_run_modes:
            continue
        
        # Extract strategy num and symbols
        strategy_num = row['strategy_num']
        symbols = row['symbols'].split(';')
        tradeP_days = row['tradeP_days'].split(';')
        # Construct the strategy configuration dictionary
        strategy_config = {
            'strategy_num': safe_int_convert(row['strategy_num']),
            'strategy_name': row['strategy_name'],
            'magic_num': magic_number_base + get_timeframe_magic_number(row['timeframe'])+ safe_int_convert(row['strategy_num']) * 1_000,
                                # base of magic number (xy_zzz_ccc)
                                # x - rev
                                # y - timeframe
                                # zzz - strategy number
                                # ccc - currency magic number - will be added at order send per the symbol
            'strategy_status': row['strategy_status'],
            'symbols': symbols,# list of symbols to trade
            'timeframe': get_mt5_timeframe(row['timeframe']),
            'tradeP_risk': safe_float_convert(row['tradeP_risk']),
            'tradeP_fixed_order_size': str_to_bool(row['tradeP_fixed_order_size']),
            'tradeP_max_trades': safe_int_convert(row['tradeP_max_trades']),
            'tradeP_hour_start': safe_int_convert(row['tradeP_hour_start']),
            'tradeP_hour_length': safe_int_convert(row['tradeP_hour_length']),
            'tradeP_days' : tradeP_days,
            'tradeP_long': str_to_bool(row['tradeP_long']),
            'tradeP_short': str_to_bool(row['tradeP_short']),
            'sl_method': row['sl_method'],
            'sl_param': safe_float_convert(row['sl_param']),
            'trail_params': {
                'trail_method': row['trail_method'],
                'trail_param': safe_float_convert(row['trail_param']),
                'trail_both_directions': str_to_bool(row['trail_both_directions']),
                'use_fast_trail': str_to_bool(row['use_fast_trail']),
                'fast_trail_minutes_count': safe_int_convert(row['fast_trail_minutes_count']),
                'fast_trail_ATR_start_multiplier': safe_float_convert(row['fast_trail_ATR_start_multiplier']),
                'fast_trail_ATR_trail_multiplier': safe_float_convert(row['fast_trail_ATR_trail_multiplier']),
                'use_move_to_breakeven': str_to_bool(row['use_move_to_breakeven']),
                'breakeven_ATRs': safe_float_convert(row['breakeven_ATRs']),
            },
            'tp_method': row['tp_method'],
            'tp_param': safe_float_convert(row['tp_param']),
            'indicators': {
                'first_indicator': {
                    'indicator_name': row['first_indicator'],
                    'indicator_use': row['first_indicator_use'],
                    'indicator_params': {
                        'a': row['first_indicator_param_a'],
                        'b': row['first_indicator_param_b'],
                        'c': row['first_indicator_param_c'],
                        'd': row['first_indicator_param_d'],
                        'e': row['first_indicator_param_e'],
                        'f': row['first_indicator_param_f'],
                        'g': row['first_indicator_param_g'],
                        'h': row['first_indicator_param_h'],
                        'i': row['first_indicator_param_i'],
                        'j': row['first_indicator_param_j'],
                        'k': row['first_indicator_param_k'],
                    },
                },
                'second_indicator': {
                    'indicator_name': row['second_indicator'],
                    'indicator_use': row['second_indicator_use'],
                    'indicator_params': {
                        'a': row['second_indicator_param_a'],
                        'b': row['second_indicator_param_b'],
                        'c': row['second_indicator_param_c'],
                        'd': row['second_indicator_param_d'],
                        'e': row['second_indicator_param_e'],
                        'f': row['second_indicator_param_f'],
                        'g': row['second_indicator_param_g'],
                        'h': row['second_indicator_param_h'],
                        'i': row['second_indicator_param_i'],
                        'j': row['second_indicator_param_j'],
                        'k': row['second_indicator_param_k'],
                    },
                },
                'third_indicator': {
                    'indicator_name': row['third_indicator'],
                    'indicator_use': row['third_indicator_use'],
                    'indicator_params': {
                        'a': row['third_indicator_param_a'],
                        'b': row['third_indicator_param_b'],
                        'c': row['third_indicator_param_c'],
                        'd': row['third_indicator_param_d'],
                        'e': row['third_indicator_param_e'],
                        'f': row['third_indicator_param_f'],
                        'g': row['third_indicator_param_g'],
                        'h': row['third_indicator_param_h'],
                        'i': row['third_indicator_param_i'],
                        'j': row['third_indicator_param_j'],
                        'k': row['third_indicator_param_k'],
                    },
                },
            },
            'exit_params': {
                'exitP_daily_profit_close': str_to_bool(row['exitP_daily_profit_close']),
                'exitP_daily_profit_close_days': safe_int_convert(row['exitP_daily_profit_close_days']),
                'exitP_daily_close': str_to_bool(row['exitP_daily_close']),
                'exitP_daily_close_days': safe_int_convert(row['exitP_daily_close_days']),
                'exitP_daily_close_hour': safe_int_convert(row['exitP_daily_close_hour']),
                'exitP_bars_close': safe_int_convert(row['exitP_bars_close']),
                'exitP_daily_candle_exit_hour': safe_int_convert(row['exitP_daily_candle_exit_hour']),
            },
            'candle_params': {
                'current_tf': {
                    'timeframe': get_mt5_timeframe(row['timeframe']),
                    'barsP_pattern': row['barsP_pattern'],
                    'barsP_pattern_count': safe_int_convert(row['barsP_pattern_count']),
                    'barsP_1st_candle': row['barsP_1st_candle'],
                    'barsP_2nd_candle': row['barsP_2nd_candle'],
                    'barsP_3rd_candle': row['barsP_3rd_candle'],
                },
                'higher_tf' : {
                    'timeframe': get_mt5_timeframe(row['barsP_higher_timeframe']),
                    'barsP_pattern': row['barsP_higher_pattern'],
                    'barsP_pattern_count': safe_int_convert(row['barsP_higher_pattern_count']),
                    'barsP_1st_candle': row['barsP_higher_1st_candle'],
                    'barsP_2nd_candle': row['barsP_higher_2nd_candle'],
                    'barsP_3rd_candle': row['barsP_higher_3rd_candle'],
                },
                'lower_tf' : {
                    'timeframe': get_mt5_timeframe(row['barsP_lower_timeframe']),
                    'barsP_pattern': row['barsP_lower_pattern'],
                    'barsP_pattern_count': safe_int_convert(row['barsP_lower_pattern_count']),
                    'barsP_1st_candle': row['barsP_lower_1st_candle'],
                    'barsP_2nd_candle': row['barsP_lower_2nd_candle'],
                    'barsP_3rd_candle': row['barsP_lower_3rd_candle'],
                },
            },
            'filterP_max_prev_prec_candle': safe_float_convert(row['filterP_max_prev_prec_candle']),
            'filterP_min_prev_prec_candle': safe_float_convert(row['filterP_min_prev_prec_candle']),
            'filterP_rsi_period': safe_float_convert(row['filterP_rsi_period']),
            'filterP_max_rsi_deviation': safe_float_convert(row['filterP_max_rsi_deviation']),
            'filterP_min_rsi_deviation': safe_float_convert(row['filterP_min_rsi_deviation']),
            'backtest_params' : {
                'backtest_start_date': safe_date_convert(row['backtest_start_date']),
                'backtest_tf' : get_mt5_timeframe(row['backtest_tf']),
            },
        }

        # Add the strategy configuration to the strategies_config dictionary
        strategies_config[strategy_num] = strategy_config
    return strategies_config




def initialize_balance_performance_file():
    """
    Initialize the performance CSV file with headers.
    If the file already exists, skip initialization.
    """
    # check if dir exsits - if not: create it
    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir, exist_ok=True)
        print(f"Created directory: {performance_dir}")

    if not os.path.exists(performance_file):
        with open(performance_file, 'w') as f:
            f.write("date,hour,open_trades,margin,balance,margin_level,equity,profit\n")
        print(f"Initialized {performance_file} with headers.")
    else:
        print(f"{performance_file} already exists. Initialization skipped.")

def write_balance_performance_file(account_info_dict, open_trades):
    """
    Write the balance and performance data to a CSV file.

    Parameters:
        account_info_dict (dict): Dictionary containing account information.
            Expected keys: 'margin', 'balance', 'margin_level', 'equity', 'profit'
        open_trades (int): Number of open trades.
    """
    if BACKTEST_MODE:
        return  # Skip writing if in backtest mode

    # Extract data from account_info_dict with default values
    margin = account_info_dict.get('margin', 0)
    balance = account_info_dict.get('balance', 0)
    margin_level = account_info_dict.get('margin_level', 0)
    equity = account_info_dict.get('equity', 0)
    profit = account_info_dict.get('profit', 0)

    # Get current date and hour
    now = time_current()
    date_str = now.strftime('%Y-%m-%d')
    hour_str = now.strftime('%H:%M:%S')

    # Get trades ammount:
    trades_num = len(open_trades)

    # Prepare the CSV line according to the header
    csv_line = f"{date_str},{hour_str},{trades_num},{margin},{balance},{margin_level},{equity},{profit}\n"

    # Append the CSV line to the performance file
    with open(performance_file, 'a') as f:
        f.write(csv_line)
    
    print_with_info(f"Appended data to {performance_file}: {csv_line.strip()}", levels_up=2)
    
    




def wait_for_new_minute(time_bar):
    """
    Wait for a new minute to start based on the TimeBar object.
    Args:
        time_bar (TimeBar): TimeBar object to track the current minute.
    """
    while not time_bar.update_tf_bar():     # wait for new minute to start
        time.sleep(1) # sleep for 1 second to wait for new bar to start
        pass
    else:
        pass
        #print("New minute started.")



def print_with_info(*args, levels_up=1, **kwargs):
    """
    Prints information about the call stack up to 'levels_up' levels.

    Parameters:
        *args: Variable length argument list to be printed after the stack info.
        levels_up (int): Number of levels up the stack to print info for. Default is 1.
        **kwargs: Arbitrary keyword arguments to be printed after the stack info.
    """
    frame = inspect.currentframe().f_back  # Start with the caller's frame
    collected_info = []
    
    for level in range(1, levels_up + 1):
        if frame:
            module = frame.f_globals.get('__name__', '<unknown module>')
            lineno = frame.f_lineno
            collected_info.append(f"Level {level}: {module} : Line {lineno}")
            frame = frame.f_back  # Move up the stack
        else:
            collected_info.append(f"Level {level}: <No further stack frames>")
    
    border = "*" * 100
    separator = "-" * 100
    print(border)
    for info in collected_info:
        print(info)
        print(separator)
    if args or kwargs:
        print(*args, **kwargs)
    print(border)



def attempt_with_stages_and_delay(
    stages,
    attempts_per_stage,
    inner_stage_delay,
    delay_between_stages,
    loop_error_msg,
    func_check_func,
    func,
    args_tuple,
):
    """
    Attempt to execute a function across multiple stages with varying delays between attempts.
    This function is useful for handling intermittent connection issues or other transient errors.

    Args:
        stages (int): Number of stages to attempt.
        attempts_per_stage (int): Number of attempts in each stage.
        inner_stage_delay (float): wait in seconds between attempts in stage
        delay_between_stages (float): wait in seconds between stages
        loop_error_msg (str): Error message to display on each failed attempt.
        func_check_func (function): Function to check the result of the function.
        func (function): Function to execute.
        args_tuple (tuple): Tuple of arguments to pass to the function.
        final_error_msg (str, optional): Error message to display if all attempts fail.

    Returns:
        Any: Result of the function execution if successful.

    Raises:
        Exception: Raises an exception with the final error message if all attempts fail.
    """
    total_attempts = 0

    for stage in range(1, stages + 1):
        for attempt in range(1, attempts_per_stage + 1):
            total_attempts += 1
            result = func(*args_tuple)
            if func_check_func(result):
                return result
            time.sleep(inner_stage_delay)
        # Increase the delay for the next stage
        print(f"{loop_error_msg} on stage {stage}, attempts so far: {total_attempts}, waiting {delay_between_stages} seconds till next stage")
        time.sleep(delay_between_stages)
    # After all attempts have failed
    return result

def catch_i_times_with_s_seconds_delay(i, s , loop_error_msg, final_error_msg, func, *args):
    """
    Attempt to execute a function a specified number of times with a delay between attempts.
    Args:
        i (int): Number of attempts to make.
        s (float): Number (or part of 1) of seconds to wait between attempts.
        func (function): Function to execute.
        loop_error_msg (str): Error message to display on each failed attempt.
        final_error_msg (str): Error message to display if all attempts fail.
        *args: Arguments to pass to the function.
    Returns:
        Any: Result of the function execution.
    """
    print("attempt_i_times_with_s_seconds_delay")
    print(f"i = {i}, s = {s}, loop_error_msg = {loop_error_msg}, final_error_msg = {final_error_msg}")
    print(f"func = {func}, args = {args}")
    for attempt in range(i):
        try:
            return func(*args)
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(s)
    return func(*args)

space = 150
hashes = 5

def print_current_time():
    current_time_str = f"current_time is: {time_current().strftime('%Y-%m-%d %H:%M:%S')}"
    spaces = (space - len(current_time_str)) // 2
    print("#"*hashes + " " * spaces + current_time_str + " " * spaces + "#"*hashes)

def print_hashtags():
    print("#" * (space + 2*hashes))

def print_hashtaged_msg(hashed_lines, *args):
    print("\n")
    print_with_info(*args, levels_up=2)
    for _ in range(hashed_lines):
        print_hashtags()
    print_current_time()
    for arg in args:
        # Ensure arg is iterable
        if isinstance(arg, dict):
            items = arg.items()
        elif isinstance(arg, (tuple, list)):
            items = arg
        else:
            items = (arg,)
        
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                key, value = item
                item_str = f"{key}: {value}"
            else:
                item_str = str(item)
            
            spaces = (space - len(item_str)) // 2
            print("#"*hashes + " " * spaces + item_str + " " * spaces + "#"*hashes)
    for _ in range(hashed_lines):
        print_hashtags()

def get_future_time(symbol, timeframe, datetime_from, num_bars):
    # Retrieve rates for the next num_bars
    rates = copy_rates_from(symbol, timeframe, datetime_from, num_bars)

    if rates is None or len(rates) < num_bars:
        print("Failed to retrieve enough bars.")

    # Extract the future time
    future_bar = rates[-1]
    future_time = datetime.fromtimestamp(future_bar['time'])
    return future_time
