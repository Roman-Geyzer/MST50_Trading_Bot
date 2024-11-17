# symbols.py
"""
This module contains the Symbol and Timeframe classes for storing symbol and timeframe data.
The Symbol class stores symbol data and initializes Timeframe instances for each timeframe.
The Timeframe class stores timeframe data and fetches historical rates for a symbol and timeframe.
Classes:
    Symbol: Class for storing symbol data, initializing Timeframe instances, and fetching rates.
    Timeframe: Class for storing timeframe data.
Methods:
    Symbol Class:
        initialize_symbols: Initialize symbols for each strategy.
        create_symbols_dict: Create a dictionary containing DataFrames for each symbol in each strategy.
        get_tf_rates: Get rates for a specific timeframe.
        check_symbol_tf_flag: Get rates for a specific timeframe.
        get_tf_obj: Get rates for a specific timeframe.
    Timeframe Class:
        calculate_tr_length: Calculate the TR length for a symbol and timeframe.
        fetch_rates: Fetch historical rates for a symbol and timeframe.
        fetch_new_bar_rates: Fetch new bar rates for all symbols and timeframes.
        get_rates: Get rates for a specific timeframe.
        get_tf_str: Get rates for a specific timeframe.
        get_symbol_str: Get rates for a specific timeframe.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from .utils import TimeBar, get_timeframe_string, attempt_with_stages_and_delay, print_hashtaged_msg, print_with_info
from .mt5_interface import TIMEFRAMES, copy_rates_from_pos
import os

# Determine if we are in backtesting mode
BACKTEST_MODE = os.environ.get('BACKTEST_MODE', 'False') == 'True'

class Symbol:
    """
    Class for storing symbol data.
    """
    def __init__(self, symbol, timeframes, strategies):
        self.symbol_str = symbol
        self.M1 = None
        self.M5 = None
        self.M15 = None
        self.M30 = None
        self.H1 = None
        self.H4 = None
        self.D1 = None
        self.W1 = None
        if TIMEFRAMES['M1'] in timeframes:
            self.M1 = Timeframe(TIMEFRAMES['M1'],self, self.symbol_str, strategies)
        if TIMEFRAMES['M5'] in timeframes:
            self.M5 = Timeframe(TIMEFRAMES['M5'],self, self.symbol_str, strategies)
        if TIMEFRAMES['M15'] in timeframes:
            self.M15 = Timeframe(TIMEFRAMES['M15'],self, self.symbol_str, strategies)
        if TIMEFRAMES['M30'] in timeframes:
            self.M30 = Timeframe(TIMEFRAMES['M30'],self, self.symbol_str, strategies)
        if TIMEFRAMES['H1'] in timeframes:
            self.H1 = Timeframe(TIMEFRAMES['H1'],self, self.symbol_str, strategies)
        if TIMEFRAMES['H4'] in timeframes:
            self.H4 = Timeframe(TIMEFRAMES['H4'],self, self.symbol_str, strategies)
        if TIMEFRAMES['D1'] in timeframes:
            self.D1 = Timeframe(TIMEFRAMES['D1'],self, self.symbol_str, strategies)
        if TIMEFRAMES['W1'] in timeframes:
            self.W1 = Timeframe(TIMEFRAMES['W1'],self, self.symbol_str, strategies)

    def __repr__(self):
        return f"Symbol({self.symbol_str}), M1: {self.M1}, M5: {self.M5}, M15: {self.M15}, M30: {self.M30}, H1: {self.H1}, H4: {self.H4}, D1: {self.D1}, W1: {self.W1}"

    def __str__(self):
        return f"Symbol({self.symbol_str}), M1: {self.M1}, M5: {self.M5}, M15: {self.M15}, M30: {self.M30}, H1: {self.H1}, H4: {self.H4}, D1: {self.D1}, W1: {self.W1}"
    
    def __eq__(self, other):
        return self.symbol_str == other.symbol_str
    
    def __hash__(self):
        return hash(self.symbol_str)
    
    def get_symbol_str(self):
        return self.symbol_str

    @staticmethod
    def initialize_symbols(strategies):
        """
        Initialize symbols for each strategy.
        Args:
            strategies (dict): Dictionary containing strategy instances.
        """
        symbols_dict = Symbol.create_symbols_dict(strategies)
        symbols = {symbol: Symbol(symbol, timeframes, strategies) for symbol, timeframes in symbols_dict.items()}
        
        return symbols
    
    @staticmethod
    def create_symbols_dict(strategies):
        """
        Initialize a dictionary containing DataFrames for each symbol in each strategy.
        Args:
            strategies (dict): Dictionary containing strategy instances.
        Returns:
            dict: Dictionary containing DataFrames for each symbol in each strategy.
        """
        symbols = {}
        for strategy in strategies.values():
            for symbol in strategy.symbols:
                if symbol not in symbols:
                    symbols[symbol] = []
                if strategy.timeframe not in symbols[symbol]:
                    symbols[symbol].append(strategy.timeframe)
                if strategy.higher_candle_patterns_active:
                    if strategy.higher_timeframe not in symbols[symbol]:
                        symbols[symbol].append(strategy.higher_timeframe)
                if strategy.lower_candle_patterns_active:
                    if strategy.lower_timeframe not in symbols[symbol]:
                        symbols[symbol].append(strategy.lower_timeframe)
                if strategy.trail_enabled: 
                    if BACKTEST_MODE: # Only add M1 if not in backtest mode - since in backtest mode we use the backtest_tf
                        if strategy.backtest_tf not in symbols[symbol]:
                            symbols[symbol].append(strategy.backtest_tf)
                    elif TIMEFRAMES['M1'] not in symbols[symbol]:
                        symbols[symbol].append(TIMEFRAMES['M1'])

        print(f"Symbols: {symbols}")
        return symbols

    def get_tf_rates(self, timeframe):
        """
        Get rates for a specific timeframe.
        Args:
            timeframe (str): Timeframe to get rates for.
        Returns:
            DataFrame: DataFrame containing rates for the symbol and timeframe.
        """
        timeframe = get_timeframe_string(timeframe)
        return getattr(self, timeframe, None).rates

    def check_symbol_tf_flag(self, timeframe):
        """
        Get rates for a specific timeframe.
        Args:
            timeframe (str): Timeframe to get rates for.
        Returns:
            DataFrame: DataFrame containing rates for the symbol and timeframe.
        """
        timeframe = get_timeframe_string(timeframe)
        return getattr(self, timeframe, None).rates_error_flag
    
    def get_tf_obj(self, timeframe):
        """
        Get rates for a specific timeframe.
        Args:
            timeframe (str): Timeframe to get rates for.
        Returns:
            DataFrame: DataFrame containing rates for the symbol and timeframe.
        """
        timeframe = get_timeframe_string(timeframe)
        return getattr(self, timeframe, None)
    
        


class Timeframe:
    """
    Class for storing timeframe data.
    """
    def __init__(self, timeframe, symbol,symbol_str, strategies):
        self.timeframe = timeframe
        self.symbol = symbol # Symbol object
        self.symbol_str = symbol_str
        self.length = self.calculate_tr_length(self.symbol_str, strategies)
        self.rates_error_flag = True
        self.rates = self.fetch_rates(symbol_str) # DF of historical rates for the symbol and timeframe

    def __repr__(self):
        return f"Timeframe({self.timeframe})"
    
    def __str__(self):
        return f"Timeframe({self.timeframe})"

    def __eq__(self, other):
        return self.timeframe == other.timeframe

    def __hash__(self):
        return hash(self.timeframe)

    
    def calculate_tr_length(self, symbol_str, strategies):
        timeframe_length_in_strategies = [12] # ATR needs minimum of 14 candles - since I'm always adding 3 candles to the max length, I can start with 12
        for strategy in strategies.values():
            if symbol_str in strategy.symbols:
                config = strategy.config
                if self.timeframe == strategy.timeframe:
                    if strategy.sl_method in ['UseCandles_SL', 'UseATR_SL']:
                        timeframe_length_in_strategies.append(strategy.sl_param)
                    if strategy.tp_method in ['UseCandles_TP', 'UseATR_TP']:
                        timeframe_length_in_strategies.append(strategy.tp_param)
                    if strategy.trail_enabled:
                        if strategy.trail_method in ['UseCandles_Trail_Close', 'UseCandles_Trail_Extreme', 'UseATR_Tral']:
                            timeframe_length_in_strategies.append(strategy.trail_param)
                        if strategy.use_fast_trail:
                            timeframe_length_in_strategies.append(strategy.fast_trail_minutes_count)

                    if strategy.first_indicator:
                        timeframe_length_in_strategies.append(strategy.config['indicators']['first_indicator']['indicator_params']['a'])
                        if strategy.second_indicator:
                            timeframe_length_in_strategies.append(strategy.config['indicators']['second_indicator']['indicator_params']['a'])
                            if strategy.third_indicator:
                                timeframe_length_in_strategies.append(strategy.config['indicators']['third_indicator']['indicator_params']['a'])
                    timeframe_length_in_strategies.append(config['filterP_rsi_period'])
                if strategy.higher_candle_patterns_active:
                    if self.timeframe == config['candle_params']['higher_tf']['timeframe']:
                        timeframe_length_in_strategies.append(strategy.config['candle_params']['higher_tf']['barsP_pattern_count'])
                if strategy.lower_candle_patterns_active:
                    if self.timeframe == config['candle_params']['lower_tf']['timeframe']:
                        timeframe_length_in_strategies.append(strategy.config['candle_params']['lower_tf']['barsP_pattern_count'])

        # Convert all elements to integers using list comprehension
        timeframe_length_in_strategies = [int(length) for length in timeframe_length_in_strategies if length  and not pd.isna(length)]
        return max(timeframe_length_in_strategies) + 3
        
    def fetch_rates(self, symbol):
        """
        Fetch historical rates for a symbol and timeframe.

        Parameters:
            symbol (str): The symbol to fetch rates for.

        Returns:
            np.recarray or None: The rates data.
        """
        self.rates_error_flag = True
        tf = self.timeframe
        length = self.length

        def check_return_func(rates):
            return rates is not None

        loop_error_msg = f"Failed to get rates for symbol: {symbol}, timeframe {tf}, length {length}"

        rates = attempt_with_stages_and_delay(
            10 , 2, 0.05, 0.5, loop_error_msg, check_return_func, copy_rates_from_pos, (symbol, tf, 0, length)
        )
        if not check_return_func(rates):
            print_hashtaged_msg(3, f"Failed to get rates for symbol: {symbol}, timeframe {tf}, length {length}")
            return None

        self.rates_error_flag = False
        return rates  # Return rates as np.recarray

    @staticmethod
    def fetch_new_bar_rates(symbols, timebar: TimeBar):
        """
        Fetch new bar rates for all symbols and timeframes.

        Args:
            symbols (dict): Dictionary of Symbol instances.
            timebar (TimeBar): Current time bar.

        Updates:
            Updates the rates for all timeframes that have a new bar.
        """
        time_frames_list = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        for symbol in symbols.values():
            for tf in time_frames_list:
                tf_obj = getattr(symbol, tf, None)
                if tf_obj  and time_frames_list.index(tf) <= time_frames_list.index(timebar.current_bar):
                    tf_obj.update_rates_if_new_bar()


    def update_rates_if_new_bar(self):
        """
        Update rates if there is a new bar since the last update.
        """
        latest_time = self.rates['time'][-1] if len(self.rates) > 0 else None
        new_rates = self.fetch_rates(self.symbol_str)

        if len(new_rates) > 0:
            new_latest_time = new_rates['time'][-1]
            if latest_time != new_latest_time:
                # New bar detected, update rates
                self.rates = new_rates
                self.rates_error_flag = False
                return
            else:
                # No new bar, do not update rates
                pass
        else:
            # Error fetching new rates
            self.rates_error_flag = True


    def get_rates(self):
        return self.rates
    
    def get_tf_str(self):
        return get_timeframe_string(self.timeframe)
    
    def get_symbol_str(self):
        return self.symbol_str