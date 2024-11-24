# indicators.py
"""
This module contains the Indicator class and its subclasses for various trading strategies.

Classes:
    Indicators: Class to manage and dispatch to specific indicator classes based on the strategy configuration.
    Indicator: Base class for all indicator classes.
    Indicator subclasses:
        NoIndicator: Class for the No Indicator strategy.
        BBIndicator: Class for the Bollinger Bands indicator.
        MAIndicator: Class for the Moving Average indicator.
        DoubleIndicator: Class for the Double pattern indicator.
        GRIndicator: Class for the Green/Red ratio indicator.
        RSIIndicator: Class for the Relative Strength Index (RSI) indicator.
        KAMAIndicator: Class for the Kaufman Adaptive Moving Average (KAMA) indicator.
        RangeIndicator: Class for Support and Resistance (SR) strategies.
        TrendIndicator: Class for trend detection strategies.
        TrendBreakoutIndicator: Class for trend breakout strategies.
        BarsTrendIndicator: Class for bar trend strategies.
        SpecialIndicator: Class for special custom indicators.
Main Functions:
    initialize_indicator: Initialize the correct indicator instance based on the strategy configuration.
    make_trade_decision: Make a trade decision based on the selected indicator class.
"""

import numpy as np
from datetime import datetime
from .constants import (TIMEFRAME_MT5_MAPPING, TRADE_DIRECTION, DEAL_TYPE, TRADE_TYPE, TRADE_DECISION)
from .utils import (safe_int_extract_from_dict, safe_float_extract_from_dict, safe_str_extract_from_dict, safe_bool_extract_from_dict,
                    print_hashtaged_msg)
from .candles import Candle
import ta as talib

class Indicators:
    def __init__(self, strategy_config):
        """
        Initialize the Indicators class based on the strategy configuration.
        This class dispatches to the specific indicator classes.

        Parameters:
            strategy_config (dict): The strategy configuration, including indicator type and parameters.
        """
        self.strategy_config = strategy_config
        self.params = strategy_config['indicator_params']

        # Mapping to indicator classes
        self.indicator_mapping = {
            '0': NoIndicator,  # No indicator
            'SR': RangeIndicator,
            'Breakout': RangeIndicator,
            'Fakeout': RangeIndicator,
            'Trend': TrendIndicator,
            'TrendBreak': TrendBreakoutIndicator,
            'Double': DoubleIndicator,
            'MACross': MAIndicator,
            'RSI_Div': RSIIndicator,
            'RSI_Div_Hidden': RSIIndicator,
            'RSI_Over': RSIIndicator,
            'RSI_With': RSIIndicator,
            'BB_With': BBIndicator,
            'BB_Return': BBIndicator,
            'BB_Over': BBIndicator,
            'GR_Ratio': GRIndicator,
            'KAMA_Cross': KAMAIndicator,
            'Bars_Trend': BarsTrendIndicator,
            'Special': SpecialIndicator,
        }

        # Initialize the appropriate indicator class based on strategy config
        self.indicator_name = self.strategy_config['indicator_name']
        self.indicator_instance = self.initialize_indicator()

    def initialize_indicator(self):
        """
        Initialize the correct indicator instance based on the strategy configuration.

        Returns:
            Indicator: An instance of the corresponding indicator class.
        """
        if self.indicator_name in self.indicator_mapping:
            indicator_class = self.indicator_mapping[self.indicator_name]
            return indicator_class(self.indicator_name, self.params)
        else:
            return NoIndicator(self.indicator_name, self.params)  # Fallback to no indicator

    def make_trade_decision(self, rates):
        """
        Make a trade decision based on the selected indicator class.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: (decision, trade_data) - where decision is a string (e.g., 'buy', 'sell', None)
                   and trade_data is a dictionary with trade parameters (entry price, stop loss, take profit).
        """
        # Check if the indicator instance exists
        if self.indicator_instance:
            return self.indicator_instance.claculuate_and_make_make_trade_decision(rates)
        else:
            return None, None
        
    def check_exit_condition(self, rates, direction):
        """
        Check if the exit condition is met based on the selected indicator class.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            direction (str): The current trade direction ('buy' or 'sell').

        Returns:
            bool: True if the exit condition is met, False otherwise.
        """
        if self.indicator_instance and self.indicator_instance.exit_decision_method:
            return self.indicator_instance.exit_decision_method(rates, direction)
        else:
            return False

class Indicator:
    def __init__(self, name, params):
        """
        Initialize the base Indicator class with common parameters.

        Parameters:
            params (dict): A dictionary of indicator parameters (common across all indicators).
        """
        self.name = name
        self.params = params
        self.trade_decision_method = None  # To be set by subclasses
        self.exit_decision_method = None  # To be set by subclasses

    def calculate_indicator_rates(self, rates):
        """
        Placeholder method to calculate indicator rates. This should be overridden by subclasses if necessary.
        """
        raise NotImplementedError("This method should be overridden by specific indicator classes.")

    def get_trade_decision_method(self):
        """
        Return the specific trade decision method for this indicator.
        This method should be overridden by subclasses.

        Returns:
            function: The trade decision method.
        """
        raise NotImplementedError("This method should be overridden by specific indicator classes.")

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Calculate the indicator and make a trade decision based on the selected trade method.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: (decision, trade_data)
        """
        if self.trade_decision_method:
            return self.trade_decision_method(rates)
        else:
            return None, None
        
    def check_and_make_exit_decision(self, rates, direction):
        """
        Check if the exit condition is met based on the selected indicator class.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            direction (str): The current trade direction ('buy' or 'sell').

        Returns:
            bool: True if the exit condition is met, False otherwise.
        """
        if self.exit_decision_method:
            return self.exit_decision_method(rates, direction)
        else:
            return False


class NoIndicator(Indicator):
    def __init__(self,name, params):
        """
        Initialize the No Indicator class with the given parameters, inherited from the Indicator superclass.

        Parameters:
            params (dict): A dictionary of parameters for the No Indicator.
        """
        super().__init__(name, params)
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on the No Indicator strategy.

        Returns:
            Tuple: (None, None)
        """
        return None, None


class BBIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Bollinger Bands Indicator with the given parameters.
        """
        super().__init__(params)
        self.bb_period = safe_int_extract_from_dict(params, 'a', 20)     # Default period of 20
        self.bb_deviation = safe_float_extract_from_dict(params, 'b', 2.0)  # Default deviation of 2.0

        # Construct label based on deviation
        deviation_int = int(self.bb_deviation * 10)
        self.bb_label = f'BB{deviation_int}'

        # Store the specific trade decision method
        self.trade_decision_method = self.get_trade_decision_method()

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator based on the parameters.
        """
        indicator_type = self.params.get('indicator_name', 'BB_With')
        decision_methods = {
            'BB_With': self.calculate_bb_with,
            'BB_Return': self.calculate_bb_return,
            'BB_Over': self.calculate_bb_over,
        }
        return decision_methods.get(indicator_type, self.calculate_bb_with)

    def calculate_bb_with(self, rates):
        """
        Determine trade decision based on BB_With strategy using precomputed boolean columns.
        """
        bool_above_col = f'{self.bb_label}_Bool_Above'
        bool_below_col = f'{self.bb_label}_Bool_Below'

        # Ensure the required columns are present
        required_cols = [bool_above_col, bool_below_col]
        if not all(col in rates.columns for col in required_cols):
            print_hashtaged_msg(1, f"Missing Bollinger Bands boolean columns in rates DataFrame: {required_cols}")
            return None, None

        bool_above = rates[bool_above_col][-1]
        bool_below = rates[bool_below_col][-1]
        current_close = rates['close'][-1]

        if bool_above:
            return 'sell', {'entry': current_close, 'sl': current_close + 10, 'tp': current_close - 20}
        elif bool_below:
            return 'buy', {'entry': current_close, 'sl': current_close - 10, 'tp': current_close + 20}
        else:
            return None, None

    def calculate_bb_return(self, rates):
        """
        Determine trade decision based on BB_Return strategy using precomputed boolean columns.
        """
        bool_above_col = f'{self.bb_label}_Bool_Above'
        bool_below_col = f'{self.bb_label}_Bool_Below'

        # Ensure the required columns are present
        required_cols = [bool_above_col, bool_below_col]
        if not all(col in rates.columns for col in required_cols):
            print_hashtaged_msg(1, f"Missing Bollinger Bands boolean columns in rates DataFrame: {required_cols}")
            return None, None

        bool_above_prev = rates[bool_above_col][-2]
        bool_below_prev = rates[bool_below_col][-2]
        bool_above_curr = rates[bool_above_col][-1]
        bool_below_curr = rates[bool_below_col][-1]
        current_close = rates['close'][-1]

        if bool_below_prev and not bool_below_curr:
            # Price was below lower band and has returned inside
            return 'buy', {'entry': current_close, 'sl': current_close - 10, 'tp': current_close + 20}
        elif bool_above_prev and not bool_above_curr:
            # Price was above upper band and has returned inside
            return 'sell', {'entry': current_close, 'sl': current_close + 10, 'tp': current_close - 20}
        else:
            return None, None

    def calculate_bb_over(self, rates):
        """
        Determine trade decision based on BB_Over strategy using precomputed middle band.
        """
        middle_band_col = f'{self.bb_label}_Middle'

        if middle_band_col not in rates.columns:
            print_hashtaged_msg(1, f"Missing Bollinger Bands middle band column in rates DataFrame: {middle_band_col}")
            return None, None

        middle_band = rates[middle_band_col][-1]
        current_close = rates['close'][-1]

        if current_close > middle_band:
            return 'buy', {'entry': current_close, 'sl': current_close - 10, 'tp': current_close + 20}
        elif current_close < middle_band:
            return 'sell', {'entry': current_close, 'sl': current_close + 10, 'tp': current_close - 20}
        else:
            return None, None


class MAIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Moving Average Indicator with the given parameters.
        """
        super().__init__(params)
        self.fast_ma_period = safe_int_extract_from_dict(params, 'a', 7)   # Fast MA period
        self.slow_ma_period = safe_int_extract_from_dict(params, 'b', 21)  # Slow MA period
        self.long_ma_period = safe_int_extract_from_dict(params, 'c', 50)  # Long MA period

        # Store the specific trade decision method
        self.trade_decision_method = self.calculate_trade_decision

    def calculate_trade_decision(self, rates):
        """
        Determine trade decision based on Moving Average strategy using precomputed columns.
        """
        # Access precomputed MA comparison columns
        fast_ma_comp_col = f'MA_{self.fast_ma_period}_comp'
        slow_ma_comp_col = f'MA_{self.slow_ma_period}_comp'
        long_ma_comp_col = f'MA_{self.long_ma_period}_comp'

        # Ensure the required columns are present
        required_cols = [fast_ma_comp_col, slow_ma_comp_col, long_ma_comp_col]
        if not all(col in rates.columns for col in required_cols):
            print_hashtaged_msg(1, f"Missing MA comparison columns in rates DataFrame: {required_cols}")
            return None, None

        fast_ma_comp = rates[fast_ma_comp_col][-1]
        slow_ma_comp = rates[slow_ma_comp_col][-1]
        long_ma_comp = rates[long_ma_comp_col][-1]

        current_close = rates['close'][-1]

        #TODO: update this method to be for crossover strategy
        #TODO: consider using the following method for MA trade filtering
        #TODO: make use of MA_7 , MA_21, and MA_50 - a column with  data on price versus higher ma - MA_7_comp -> can get 1 of 3 string values: 'above' , 'below' , 'equal'  
        if fast_ma_comp == 'above' and slow_ma_comp == 'above' and long_ma_comp == 'above':
            # All MAs are below the price, indicating upward trend
            return 'buy', {'entry': current_close, 'sl': current_close - 10, 'tp': current_close + 20}
        elif fast_ma_comp == 'below' and slow_ma_comp == 'below' and long_ma_comp == 'below':
            # All MAs are above the price, indicating downward trend
            return 'sell', {'entry': current_close, 'sl': current_close + 10, 'tp': current_close - 20}
        else:
            return None, None



class GRIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the GR Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for GR Ratio calculation.
        """
        super().__init__(params)
        self.ratio_candles_count = safe_int_extract_from_dict(params, 'a', 100)  # Number of candles to calculate ratio
        self.buy_enter_limit = safe_float_extract_from_dict(params, 'b', 1.35)  # Buy entry limit for GR ratio
        self.sell_enter_limit = safe_float_extract_from_dict(params, 'c', 1.35)  # Sell entry limit for GR ratio
        self.buy_exit_limit = safe_float_extract_from_dict(params, 'd', 1.0)  # Buy exit limit for GR ratio
        self.sell_exit_limit = safe_float_extract_from_dict(params, 'e', 1.0)  # Sell exit limit for GR ratio

        # Store the specific trade decision method
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision


    def calculate_and_make_trade_decision(self, rates):
        """
        Make a trade decision based on the precomputed GR ratio.

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data)
        """
        ga_col = f'GA_{self.ratio_candles_count}'

        if ga_col not in rates.columns:
            print_hashtaged_msg(1, f"Missing GR ratio column in rates DataFrame: {ga_col}")
            return None, None

        gr_ratio = rates[ga_col][-1]

        current_close = rates['close'][-1]
        current_low = rates['low'][-1]
        current_high = rates['high'][-1]

        # If GR ratio signals a buy
        if gr_ratio > self.buy_enter_limit:
            return 'buy', {'entry': current_close, 'sl': current_low - 10, 'tp': current_close + 20}
        # If GR ratio signals a sell
        elif gr_ratio < self.sell_enter_limit:
            return 'sell', {'entry': current_close, 'sl': current_high + 10, 'tp': current_close - 20}
        else:
            return None, None



class RSIIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the RSI Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for RSI calculation.
        """
        super().__init__(params)
        self.rsi_period = safe_int_extract_from_dict(params, 'a', 14)  # RSI period
        self.rsi_over_extended = safe_float_extract_from_dict(params, 'b', 20.0)
        self.rsi_div_lookback_period = safe_int_extract_from_dict(params, 'c', 50)

        # Mapping for RSI trade decision methods
        self.decision_methods = {
            'RSI_Div': self.calculate_rsi_div,
            'RSI_Div_Hidden': self.calculate_rsi_div_hidden,
            'RSI_Over': self.calculate_rsi_over,
            'RSI_With': self.calculate_rsi_with
        }

        # Store the specific trade decision method based on the parameters
        self.trade_decision_method = self.get_trade_decision_method()

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator.

        Returns:
            function: The trade decision method.
        """
        indicator_type = self.params.get('type', 'RSI_Div')
        return self.decision_methods.get(indicator_type, self.calculate_rsi_div)

    def calculate_rsi_div(self, rates):
        """
        Determine trade decision based on RSI divergence strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        if 'RSI' not in rates.columns:
            print_hashtaged_msg(1, "Missing RSI column in rates DataFrame")
            return None, None

        self.rsi_values = rates['RSI'].values
        if len(self.rsi_values) < self.rsi_period:
            return None, None

        if self.rsi_divergence_check_is_buy(rates):
            return 'buy', None
        elif self.rsi_divergence_check_is_sell(rates):
            return 'sell', None
        return None, None

    def calculate_rsi_div_hidden(self, rates):
        """
        Determine trade decision based on hidden RSI divergence strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        if 'RSI' not in rates.columns:
            print_hashtaged_msg(1, "Missing RSI column in rates DataFrame")
            return None, None

        self.rsi_values = rates['RSI'].values
        if len(self.rsi_values) < self.rsi_period:
            return None, None

        if self.rsi_hidden_divergence_check_is_buy(rates):
            return 'buy', None
        elif self.rsi_hidden_divergence_check_is_sell(rates):
            return 'sell', None
        return None, None

    def calculate_rsi_over(self, rates):
        """
        Determine trade decision based on RSI overbought/oversold levels.

        Returns:
            Tuple: (decision, trade_data)
        """
        if 'RSI' not in rates.columns:
            print_hashtaged_msg(1, "Missing RSI column in rates DataFrame")
            return None, None

        self.rsi_values = rates['RSI'].values
        if len(self.rsi_values) < self.rsi_period:
            return None, None

        if self.rsi_oversold_check_is_buy():
            return 'buy', None
        elif self.rsi_overbought_check_is_sell():
            return 'sell', None
        return None, None

    def calculate_rsi_with(self, rates):
        """
        Determine trade decision based on RSI trending with overbought/oversold levels.

        Returns:
            Tuple: (decision, trade_data)
        """
        if 'RSI' not in rates.columns:
            print_hashtaged_msg(1, "Missing RSI column in rates DataFrame")
            return None, None

        self.rsi_values = rates['RSI'].values
        if len(self.rsi_values) < self.rsi_period:
            return None, None

        if self.rsi_with_trend_check_is_buy():
            return 'buy', None
        elif self.rsi_with_trend_check_is_sell():
            return 'sell', None
        return None, None

    def rsi_divergence_check_is_buy(self, rates):
        """
        Check for a bullish RSI divergence.

        Parameters:
            rates (DataFrame): Historical price data (OHLC).

        Returns:
            bool: True if a bullish divergence is detected, False otherwise.
        """
        idx = -3  # Current index
        if not self.is_local_rsi_min(idx):
            return False

        for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
            if abs(i) >= len(self.rsi_values):
                break
            if self.rsi_values[i] < 50 - self.rsi_over_extended:
                if self.rsi_values[i] < self.rsi_values[idx]:
                    if self.is_local_rsi_min(i):
                        if rates['close'][i] > rates['close'][idx]:
                            return True
        return False

    def rsi_divergence_check_is_sell(self, rates):
        """
        Check for a bearish RSI divergence.

        Parameters:
            rates (DataFrame): Historical price data (OHLC).

        Returns:
            bool: True if a bearish divergence is detected, False otherwise.
        """
        idx = -3  # Current index
        if not self.is_local_rsi_max(idx):
            return False

        for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
            if abs(i) >= len(self.rsi_values):
                break
            if self.rsi_values[i] > 50 + self.rsi_over_extended:
                if self.rsi_values[i] > self.rsi_values[idx]:
                    if self.is_local_rsi_max(i):
                        if rates['close'][i] < rates['close'][idx]:
                            return True
        return False

    def rsi_hidden_divergence_check_is_buy(self, rates):
        """
        Check for a hidden bullish RSI divergence.

        Parameters:
            rates (DataFrame): Historical price data (OHLC).

        Returns:
            bool: True if a hidden bullish divergence is detected, False otherwise.
        """
        idx = -3
        if not self.is_local_rsi_min(idx):
            return False
        if self.rsi_values[idx] < 50 - self.rsi_over_extended:
            for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
                if abs(i) >= len(self.rsi_values):
                    break
                if self.rsi_values[i] > self.rsi_values[idx]:
                    if self.is_local_rsi_min(i):
                        if rates['close'][i] < rates['close'][idx]:
                            return True
        return False

    def rsi_hidden_divergence_check_is_sell(self, rates):
        """
        Check for a hidden bearish RSI divergence.

        Parameters:
            rates (DataFrame): Historical price data (OHLC).

        Returns:
            bool: True if a hidden bearish divergence is detected, False otherwise.
        """
        idx = -3
        if not self.is_local_rsi_max(idx):
            return False
        if self.rsi_values[idx] > 50 + self.rsi_over_extended:
            for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
                if abs(i) >= len(self.rsi_values):
                    break
                if self.rsi_values[i] < self.rsi_values[idx]:
                    if self.is_local_rsi_max(i):
                        if rates['close'][i] > rates['close'][idx]:
                            return True
        return False

    def rsi_overbought_check_is_sell(self):
        """
        Check if the RSI indicates overbought conditions (sell signal).
        """
        idx = -3
        return self.rsi_values[idx] > 50 + self.rsi_over_extended and self.is_local_rsi_max(idx)

    def rsi_oversold_check_is_buy(self):
        """
        Check if the RSI indicates oversold conditions (buy signal).
        """
        idx = -3
        return self.rsi_values[idx] < 50 - self.rsi_over_extended and self.is_local_rsi_min(idx)

    def rsi_with_trend_check_is_buy(self):
        """
        Check if RSI indicates a buy signal when trending with oversold conditions.
        """
        return self.rsi_values[-1] > 50 + self.rsi_over_extended

    def rsi_with_trend_check_is_sell(self):
        """
        Check if RSI indicates a sell signal when trending with overbought conditions.
        """
        return self.rsi_values[-1] < 50 - self.rsi_over_extended

    def is_local_rsi_max(self, idx):
        """
        Check if the RSI value at a given index is a local maximum.

        Parameters:
            idx (int): Index to check for local max.

        Returns:
            bool: True if the RSI is a local max, False otherwise.
        """
        try:
            return (
                self.rsi_values[idx] > self.rsi_values[idx + 1] and self.rsi_values[idx] > self.rsi_values[idx + 2] and
                self.rsi_values[idx] > self.rsi_values[idx - 1] and self.rsi_values[idx] > self.rsi_values[idx - 2]
            )
        except IndexError:
            return False

    def is_local_rsi_min(self, idx):
        """
        Check if the RSI value at a given index is a local minimum.

        Parameters:
            idx (int): Index to check for local min.

        Returns:
            bool: True if the RSI is a local min, False otherwise.
        """
        try:
            return (
                self.rsi_values[idx] < self.rsi_values[idx + 1] and self.rsi_values[idx] < self.rsi_values[idx + 2] and
                self.rsi_values[idx] < self.rsi_values[idx - 1] and self.rsi_values[idx] < self.rsi_values[idx - 2]
            )
        except IndexError:
            return False

class DoubleIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Double Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for Double pattern detection.
        """
        super().__init__(params)
        self.period_for_double = safe_int_extract_from_dict(params, 'a', 100)  # Lookback period for double
        self.slack_perc = safe_float_extract_from_dict(params, 'b', 0.1)  # Slack for double percentage
        self.wait_between_candles = safe_int_extract_from_dict(params, 'c', 6)  # Min candles between double tops/bottoms
        self.max_bars_from_double = safe_int_extract_from_dict(params, 'd', 5)  # Max bars after double to trigger a trade
        self.max_distance_perc = safe_float_extract_from_dict(params, 'e', 3.0)  # Max distance from double percentage
        self.min_distance_perc = safe_float_extract_from_dict(params, 'f', 0.1)  # Min distance from double percentage

        # Initialize state variables
        self.double_up = False
        self.double_down = False
        self.double_up_price = 0.0
        self.double_down_price = 0.0
        self.first_touch_value = 0.0
        self.second_touch_value = 0.0

        # Set the trade decision method
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    def find_extreme(self, rates, period: int, mode: str):
        """
        Helper function to find the minimum or maximum in the given period.

        Parameters:
            rates (np.recarray): Historical rates.
            period (int): Lookback period.
            mode (str): Either 'low' for lowest points or 'high' for highest points.

        Returns:
            int: Index of the min or max value in the rates array.
        """
        if mode == 'low':
            idx = np.argmin(rates['low'][-period:])
        elif mode == 'high':
            idx = np.argmax(rates['high'][-period:])
        else:
            print_hashtaged_msg(1, "Invalid mode:", mode)
            return None
        return len(rates) - period + idx  # Convert to index in rates

    def calculate_double_up(self, rates):
        """
        Identify double bottom (up) pattern and trigger a potential buy signal.
        """
        idx1 = self.find_extreme(rates, self.period_for_double, 'low')
        idx2 = self.find_extreme(rates, self.period_for_double - 1, 'low')

        if idx1 is None or idx2 is None:
            return

        # Calculate the difference in candle indices
        candle_diff = abs(idx1 - idx2)
        if candle_diff < self.wait_between_candles:
            return

        self.first_touch_value = rates['low'][idx1]
        self.second_touch_value = rates['low'][idx2]

        slack_threshold = self.slack_perc * rates['close'][-1] / 100
        if abs(self.first_touch_value - self.second_touch_value) > slack_threshold:
            return

        if rates['open'][-1] < self.second_touch_value:
            return

        self.double_up = True
        self.double_up_price = self.second_touch_value

    def calculate_double_down(self, rates):
        """
        Identify double top (down) pattern and trigger a potential sell signal.
        """
        idx1 = self.find_extreme(rates, self.period_for_double, 'high')
        idx2 = self.find_extreme(rates, self.period_for_double - 1, 'high')

        if idx1 is None or idx2 is None:
            return

        # Calculate the difference in candle indices
        candle_diff = abs(idx1 - idx2)
        if candle_diff < self.wait_between_candles:
            return

        self.first_touch_value = rates['high'][idx1]
        self.second_touch_value = rates['high'][idx2]

        slack_threshold = self.slack_perc * rates['close'][-1] / 100
        if abs(self.first_touch_value - self.second_touch_value) > slack_threshold:
            return

        if rates['open'][-1] > self.second_touch_value:
            return

        self.double_down = True
        self.double_down_price = self.second_touch_value

    def double_check_is_buy(self) -> bool:
        """
        Check if the Double Up pattern is valid for a buy signal.
        """
        return self.double_up

    def double_check_is_sell(self) -> bool:
        """
        Check if the Double Down pattern is valid for a sell signal.
        """
        return self.double_down

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make the trade decision based on the double top or bottom pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        self.calculate_double_up(rates)
        self.calculate_double_down(rates)

        current_close = rates['close'][-1]

        if self.double_check_is_buy():
            trade_data = {
                'entry': current_close,
                'sl': self.double_up_price - 10,  # Adjust SL as needed
                'tp': current_close + 20          # Adjust TP as needed
            }
            self.reset_signals()
            return 'buy', trade_data

        if self.double_check_is_sell():
            trade_data = {
                'entry': current_close,
                'sl': self.double_down_price + 10,  # Adjust SL as needed
                'tp': current_close - 20           # Adjust TP as needed
            }
            self.reset_signals()
            return 'sell', trade_data

        return None, None

    def reset_signals(self):
        """
        Reset the double up/down signals after a trade is executed or conditions change.
        """
        self.double_up = False
        self.double_down = False
        self.double_up_price = 0.0
        self.double_down_price = 0.0
        self.first_touch_value = 0.0
        self.second_touch_value = 0.0



class KAMAIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the KAMA Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for KAMA calculation.
        """
        super().__init__(params)
        self.er_candles_count = safe_int_extract_from_dict(params, 'a', 20)  # Number of candles for calculating ER
        self.fast_ema_period = 2
        self.slow_ema_period = 30
        self.sc_fast = 2 / (self.fast_ema_period + 1)
        self.sc_slow = 2 / (self.slow_ema_period + 1)

        # Initialize KAMA array
        self.kama = None

        # Store the specific trade decision method
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    def calculate_kama(self, rates):
        """
        Calculate the KAMA values.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            np.ndarray: KAMA values.
        """
        close_prices = rates['close']
        if len(close_prices) < self.er_candles_count:
            return None

        # Calculate Efficiency Ratio (ER)
        change = np.abs(close_prices - np.roll(close_prices, self.er_candles_count))
        volatility = np.sum(np.abs(close_prices - np.roll(close_prices, 1)), axis=0)
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility != 0)

        # Smoothing Constant (SC)
        sc = (er * (self.sc_fast - self.sc_slow) + self.sc_slow) ** 2

        # Initialize KAMA
        kama = np.copy(close_prices)
        for i in range(self.er_candles_count + 1, len(close_prices)):
            kama[i] = kama[i - 1] + sc[i] * (close_prices[i] - kama[i - 1])

        return kama

    def kama_check_is_buy(self, rates):
        """
        Check if the KAMA indicator signals a buy.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if KAMA indicates a buy, False otherwise.
        """
        if self.kama is None:
            self.kama = self.calculate_kama(rates)
            if self.kama is None:
                return False

        return rates['close'][-1] > self.kama[-1] and rates['close'][-2] < self.kama[-2]

    def kama_check_is_sell(self, rates):
        """
        Check if the KAMA indicator signals a sell.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if KAMA indicates a sell, False otherwise.
        """
        if self.kama is None:
            self.kama = self.calculate_kama(rates)
            if self.kama is None:
                return False

        return rates['close'][-1] < self.kama[-1] and rates['close'][-2] > self.kama[-2]

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on KAMA signals.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        if self.kama_check_is_buy(rates):
            return 'buy', {'entry': rates['close'][-1], 'sl': rates['low'][-1] - 10, 'tp': rates['close'][-1] + 20}
        if self.kama_check_is_sell(rates):
            return 'sell', {'entry': rates['close'][-1], 'sl': rates['high'][-1] + 10, 'tp': rates['close'][-1] - 20}
        return None, None


class RangeIndicator(Indicator):
    def __init__(self, name, params):
        """
        Initialize the Range Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for SR, Breakout, and Fakeout calculations.
        """
        super().__init__(name, params)
        self.period_for_sr = safe_int_extract_from_dict(params, 'a', 100)
        self.touches_for_sr = safe_int_extract_from_dict(params, 'b', 3)
        self.slack_for_sr_atr_div = safe_float_extract_from_dict(params, 'c', 10.0)
        self.atr_rejection_multiplier = safe_float_extract_from_dict(params, 'd', 1.0)
        self.max_distance_from_sr_atr = safe_float_extract_from_dict(params, 'e', 2.0)
        self.min_height_of_sr_distance = safe_float_extract_from_dict(params, 'f', 3.0)
        self.max_height_of_sr_distance = safe_float_extract_from_dict(params, 'g', 200.0)

        self.slack_for_breakout_atr = safe_float_extract_from_dict(params, 'h', 0.1)
        
        self.bars_from_fakeout = safe_int_extract_from_dict(params, 'i', 2)
        self.bars_before_fakeout = safe_int_extract_from_dict(params, 'j', 2)
        self.fakeout_atr_slack = safe_float_extract_from_dict(params, 'k', 0.5)


        # Initialize SR levels
        self.upper_sr = 0.0
        self.lower_sr = 0.0
        self.prev_upper_sr_level = 0.0
        self.prev_lower_sr_level = 0.0
        self.upper_limit = 0.0
        self.lower_limit = 0.0

        # Determine the strategy type
        self.strategy_type = self.params.get('type', 'SR')

        # Store the specific trade decision method based on the parameters
        self.trade_decision_method = self.get_trade_decision_method()
        self.exit_decision_method = self.get_exit_decision_method()

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator based on the parameters.
        Returns:
            function: The trade decision method.
        """
        # Mapping for trade decision methods
        decision_methods = {
            'SR': self.sr_trade_decision,
            'Breakout': self.breakout_trade_decision,
            'Fakeout': self.fakeout_trade_decision
        }
        return decision_methods.get(self.name)

    def get_exit_decision_method(self):
        """
        Return the appropriate exit decision method for this indicator based on the parameters.
        Returns:
            function: The exit decision method.
        """
        # Mapping for exit decision methods
        exit_methods = {
            'SR': self.sr_exit_decision,
            'Breakout': self.breakout_exit_decision,
            'Fakeout': self.fakeout_exit_decision
        }
        return exit_methods.get(self.name)

    def calculate_sr_levels(self, rates):
        """
        Calculate the Support and Resistance (SR) levels for the given rates.

        Parameters:
            rates (DataFrame): Historical price data (OHLC).
        """
        if len(rates) < self.period_for_sr:
            return

        recent_rates = rates[-self.period_for_sr:]

        # Initialize uSlackForSR and uRejectionFromSR
        atr = rates['ATR'][-1]
        uSlackForSR = atr / self.slack_for_sr_atr_div
        uRejectionFromSR = atr * self.atr_rejection_multiplier

        current_open = rates['open'][-1]

        # Initialize HighSR and LowSR
        HighSR = current_open + self.min_height_of_sr_distance * uSlackForSR
        LowSR = current_open - self.min_height_of_sr_distance * uSlackForSR

        # LocalMax and LocalMin
        LocalMax = np.max(recent_rates['high'])
        LocalMin = np.min(recent_rates['low'])

        # Initialize LoopCounter
        LoopCounter = 0

        # Upper SR Level
        while LoopCounter < self.max_height_of_sr_distance:
            UpperSR = HighSR
            num_touches = self.count_touches(UpperSR, recent_rates, uRejectionFromSR, upper=True)
            if num_touches >= self.touches_for_sr:
                self.upper_sr = UpperSR
                break
            else:
                HighSR += uSlackForSR
                LoopCounter += 1
                if HighSR > LocalMax:
                    self.upper_sr = 0
                    self.upper_limit = HighSR
                    break

        # Reset LoopCounter for LowerSR
        LoopCounter = 0

        # Lower SR Level
        while LoopCounter < self.max_height_of_sr_distance:
            LowerSR = LowSR
            num_touches = self.count_touches(LowerSR, recent_rates, uRejectionFromSR, upper=False)
            if num_touches >= self.touches_for_sr:
                self.lower_sr = LowerSR
                break
            else:
                LowSR -= uSlackForSR
                LoopCounter += 1
                if LowSR < LocalMin:
                    self.lower_sr = 0
                    self.lower_limit = LowSR
                    break

        # Store previous SR levels for Breakout and Fakeout checks
        self.prev_upper_sr_level = self.upper_sr
        self.prev_lower_sr_level = self.lower_sr

    def count_touches(self, current_hline, recent_rates, uRejectionFromSR, upper=True):
        """
        Count the number of touches to the given SR level.

        Parameters:
            current_hline (float): The SR level to check.
            recent_rates (DataFrame): The recent rates to check.
            uRejectionFromSR (float): The rejection slack based on ATR.
            upper (bool): True if checking for upper SR, False for lower SR.

        Returns:
            int: Number of touches.
        """
        counter = 0
        for idx in range(len(recent_rates) - 1):
            open_price = recent_rates['open'][idx]
            close_price = recent_rates['close'][idx]
            high_price = recent_rates['high'][idx]
            low_price = recent_rates['low'][idx]
            candle_size = abs(high_price - low_price)

            if upper:
                # Upper SR check
                if open_price < current_hline and close_price < current_hline:
                    if high_price > current_hline or (
                        candle_size > uRejectionFromSR and (current_hline - high_price) < uRejectionFromSR / 2
                    ):
                        counter += 1
            else:
                # Lower SR check
                if open_price > current_hline and close_price > current_hline:
                    if low_price < current_hline or (
                        candle_size > uRejectionFromSR and (low_price - current_hline) < uRejectionFromSR / 2
                    ):
                        counter += 1
        return counter

    def sr_trade_decision(self, rates):
        """
        Make a trade decision based on SR strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_sr_levels(rates)
        current_open = rates['open'][-1]
        current_close = rates['close'][-1]
        atr = rates['ATR'][-1]

        # For buy signal
        if self.lower_sr != 0:
            sr_level = self.lower_sr
            if (current_open - sr_level) < self.max_distance_from_sr_atr * atr:
                # Buy signal
                return 'buy', {'entry': current_close, 'sl': sr_level - 10, 'tp': self.upper_sr}

        # For sell signal
        if self.upper_sr != 0:
            sr_level = self.upper_sr
            if (sr_level - current_open) < self.max_distance_from_sr_atr * atr:
                # Sell signal
                return 'sell', {'entry': current_close, 'sl': sr_level + 10, 'tp': self.lower_sr}

        return None, None

    def sr_exit_decision(self, rates, direction):
        """
        Return an exit decision based on SR strategy.

        Returns:
            Bool: True if the trade should be closed, False otherwise.
        """
        sr_trade, _ = self.sr_trade_decision(rates)
        if sr_trade == 'buy' and direction == 'sell':
            return True  # Close the trade
        elif sr_trade == 'sell' and direction == 'buy':
            return True  # Close the trade
        return False  # Do not close the trade

    def breakout_trade_decision(self, rates):
        """
        Make a trade decision based on Breakout strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_sr_levels(rates)
        current_close = rates['close'][-1]
        previous_close = rates['close'][-2]
        atr = rates['ATR'][-1]

        # For buy signal
        if self.prev_upper_sr_level != 0:
            sr_level = self.prev_upper_sr_level
            if previous_close > sr_level + self.slack_for_breakout_atr * atr:
                # Buy signal
                return 'buy', {'entry': current_close, 'sl': self.prev_lower_sr_level - 10, 'tp': current_close + 20}

        # For sell signal
        if self.prev_lower_sr_level != 0:
            sr_level = self.prev_lower_sr_level
            if previous_close < sr_level - self.slack_for_breakout_atr * atr:
                # Sell signal
                return 'sell', {'entry': current_close, 'sl': self.prev_upper_sr_level + 10, 'tp': current_close - 20}

        return None, None

    def breakout_exit_decision(self, rates, direction):
        """
        Make an exit decision based on Breakout strategy.

        Returns:
            Bool: True if the trade should be closed, False otherwise.
        """
        breakout_trade, _ = self.breakout_trade_decision(rates)
        if breakout_trade == 'buy' and direction == 'sell':
            return True
        elif breakout_trade == 'sell' and direction == 'buy':
            return True
        return False

    def fakeout_trade_decision(self, rates):
        """
        Make a trade decision based on Fakeout strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_sr_levels(rates)
        atr = rates['ATR'][-1]

        total_bars_needed = self.bars_from_fakeout + self.bars_before_fakeout
        if len(rates) < total_bars_needed + 1:
            return None, None

        # For buy signal
        if self.lower_sr != 0:
            sr_level = self.lower_sr

            # Get lows for fakeout and previous period
            fakeout_lows = rates['low'][-self.bars_from_fakeout:]
            previous_lows = rates['low'][-(total_bars_needed):-self.bars_from_fakeout]

            low_in_fakeout = np.min(fakeout_lows)
            low_before_fakeout = np.min(previous_lows)

            if low_in_fakeout <= sr_level - self.fakeout_atr_slack * atr:
                if low_before_fakeout >= sr_level:
                    sr_decision, _ = self.sr_trade_decision(rates)
                    if sr_decision == 'buy':
                        return 'buy', {'entry': rates['close'][-1], 'sl': sr_level - 10, 'tp': self.upper_sr}

        # For sell signal
        if self.upper_sr != 0:
            sr_level = self.upper_sr

            fakeout_highs = rates['high'][-self.bars_from_fakeout:]
            previous_highs = rates['high'][-(total_bars_needed):-self.bars_from_fakeout]

            high_in_fakeout = np.max(fakeout_highs)
            high_before_fakeout = np.max(previous_highs)

            if high_in_fakeout >= sr_level + self.fakeout_atr_slack * atr:
                if high_before_fakeout <= sr_level:
                    sr_decision, _ = self.sr_trade_decision(rates)
                    if sr_decision == 'sell':
                        return 'sell', {'entry': rates['close'][-1], 'sl': sr_level + 10, 'tp': self.lower_sr}

        return None, None

    def fakeout_exit_decision(self, rates, direction):
        """
        Make an exit decision based on Fakeout strategy.

        Returns:
            Bool: True if the trade should be closed, False otherwise.
        """
        fakeout_trade, _ = self.fakeout_trade_decision(rates)
        if fakeout_trade == 'buy' and direction == 'sell':
            return True  # Close the trade
        elif fakeout_trade == 'sell' and direction == 'buy':
            return True  # Close the trade
        return False  # Do not close the trade

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on SR, Breakout, or Fakeout conditions.

        Parameters:
            rates (DataFrame): Historical price data (OHLC).

        Returns:
            Tuple: (decision, trade_data)
        """
        return self.trade_decision_method(rates)


class TrendIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Trend Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for trendline calculation.
        """
        super().__init__(params)
        self.period_for_trend = safe_int_extract_from_dict(params, 'a', 100)
        self.slack_for_trend_perc = safe_float_extract_from_dict(params, 'b', 0.1)
        self.min_slope_perc_div = safe_float_extract_from_dict(params, 'c', 10.0)
        self.max_distance_from_trend_perc = safe_float_extract_from_dict(params, 'd', 2.0)

        # Store the specific trade decision method
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    def detect_trend(self, rates):
        """
        Detect upward or downward trend in the historical price data.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: (slope, intercept) or None
        """
        if len(rates) < self.period_for_trend:
            return None

        prices = rates['close'][-self.period_for_trend:]
        x = np.arange(len(prices))
        coef = np.polyfit(x, prices, 1)
        slope = coef[0]
        intercept = coef[1]

        return slope, intercept

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on trendline conditions.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: (decision, trade_data)
        """
        trend = self.detect_trend(rates)
        if trend is None:
            return None, None

        slope, intercept = trend
        current_price = rates['close'][-1]
        x = len(rates['close']) - 1
        trend_value = slope * x + intercept
        distance_from_trend = abs(current_price - trend_value)

        if slope > 0 and distance_from_trend < (self.max_distance_from_trend_perc / 100) * current_price:
            # Upward trend detected, buy signal
            return 'buy', {'entry': current_price, 'sl': current_price - 10, 'tp': current_price + 20}
        elif slope < 0 and distance_from_trend < (self.max_distance_from_trend_perc / 100) * current_price:
            # Downward trend detected, sell signal
            return 'sell', {'entry': current_price, 'sl': current_price + 10, 'tp': current_price - 20}
        return None, None

class TrendBreakoutIndicator(TrendIndicator):
    def __init__(self, params):
        """
        Initialize the Trend Breakout Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for trendline and breakout calculation.
        """
        super().__init__(params)
        self.slack_for_breakout = safe_float_extract_from_dict(params, 'e', 0.1)
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on trendline breakouts.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: (decision, trade_data)
        """
        trend = self.detect_trend(rates)
        if trend is None:
            return None, None

        slope, intercept = trend
        current_price = rates['close'][-1]
        previous_price = rates['close'][-2]
        x_current = len(rates['close']) - 1
        x_previous = x_current - 1
        trend_value_current = slope * x_current + intercept
        trend_value_previous = slope * x_previous + intercept

        if slope < 0 and previous_price > trend_value_previous + (self.slack_for_breakout / 100) * previous_price:
            # Breakout above downtrend, buy signal
            return 'buy', {'entry': current_price, 'sl': current_price - 10, 'tp': current_price + 20}
        elif slope > 0 and previous_price < trend_value_previous - (self.slack_for_breakout / 100) * previous_price:
            # Breakout below uptrend, sell signal
            return 'sell', {'entry': current_price, 'sl': current_price + 10, 'tp': current_price - 20}
        return None, None

class BarsTrendIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Bars Trend Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for bar trend calculation.
        """
        super().__init__(params)
        self.bars_lookback_period = safe_int_extract_from_dict(params, 'a', 10)
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on highest/lowest bars in a trend.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: (decision, trade_data)
        """
        if len(rates) < self.bars_lookback_period + 1:
            return None, None

        recent_rates = rates[-(self.bars_lookback_period + 1):]

        lowest_close = np.min(recent_rates['close'][:-1])
        highest_close = np.max(recent_rates['close'][:-1])
        current_close = recent_rates['close'][-1]

        if current_close < lowest_close:
            return 'buy', {'entry': current_close, 'sl': lowest_close - 10, 'tp': highest_close}
        elif current_close > highest_close:
            return 'sell', {'entry': current_close, 'sl': highest_close + 10, 'tp': lowest_close}
        return None, None

class SpecialIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the SpecialIndicator with parameters.

        Parameters:
            params (dict): A dictionary containing parameters for the Special Indicator.
        """
        super().__init__(params)
        self.special_id = safe_int_extract_from_dict(params, 'a', 1)
        self.xbars_special = safe_int_extract_from_dict(params, 'b', 5)
        self.xbars_2nd_special = safe_int_extract_from_dict(params, 'c', 5)
        self.special_period = safe_int_extract_from_dict(params, 'd', 3)
        self.special_multiplier_entry = safe_float_extract_from_dict(params, 'e', 0.01)
        self.special_multiplier_sl = safe_float_extract_from_dict(params, 'f', 2.0)
        self.trade_decision_method = self.calculate_trade_decision

    def calculate_trade_decision(self, rates):
        """
        Determine trade decision based on the Special indicator strategy.

        Parameters:
            rates (np.recarray): The OHLC data.

        Returns:
            Tuple: (decision, trade_data)
        """
        # Dynamically call the appropriate method based on special_id
        method_name = f"special_{self.special_id}_calculate_trade_decision"
        method = getattr(self, method_name, None)
        if callable(method):
            return method(rates)
        return None, None

    def special_100_calculate_trade_decision(self, rates):
        """
        Special Strategy 100: Similar to BarsTrendIndicator.

        Returns:
            Tuple: (decision, trade_data)
        """
        return BarsTrendIndicator({'a': self.xbars_special}).claculuate_and_make_make_trade_decision(rates)

    def special_101_calculate_trade_decision(self, rates):
        """
        Special Strategy 101: Inside Bar Strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        if len(rates) < 3:
            return None, None

        current_bar = rates[-1]
        previous_bar = rates[-2]
        third_bar = rates[-3]

        if current_bar['high'] < previous_bar['high'] and current_bar['low'] > third_bar['low']:
            candle_size = abs(current_bar['high'] - current_bar['low'])
            return 'buy', {'entry': current_bar['close'], 'sl': current_bar['low'] - candle_size, 'tp': None}
        return None, None

    def special_102_calculate_trade_decision(self, rates):
        """
        Special Strategy 102: Price deviation from long MA.

        Returns:
            Tuple: (decision, trade_data)
        """
        if len(rates) < self.special_period:
            return None, None

        close_prices = rates['close']
        long_ma = np.mean(close_prices[-self.special_period:])

        current_close = close_prices[-1]

        if current_close < (1 - self.special_multiplier_entry) * long_ma:
            return 'buy', {'entry': current_close, 'sl': current_close - self.special_multiplier_sl * long_ma, 'tp': None}
        elif current_close > (1 + self.special_multiplier_entry) * long_ma:
            return 'sell', {'entry': current_close, 'sl': current_close + self.special_multiplier_sl * long_ma, 'tp': None}
        return None, None

    def special_103_calculate_trade_decision(self, rates):
        """
        Special Strategy 103: Consecutive higher highs or lower lows.

        Returns:
            Tuple: (decision, trade_data)
        """
        if len(rates) < self.xbars_special + 2:
            return None, None

        recent_rates = rates[-(self.xbars_special + 2):]

        # Check for higher highs and higher closes
        hhhc = all(recent_rates['close'][i] > recent_rates['close'][i - 1] and recent_rates['high'][i] > recent_rates['high'][i - 1]
                   for i in range(1, self.xbars_special + 1))

        # Check for lower lows and lower closes
        lllc = all(recent_rates['close'][i] < recent_rates['close'][i - 1] and recent_rates['low'][i] < recent_rates['low'][i - 1]
                   for i in range(1, self.xbars_special + 1))

        if lllc:
            return 'buy', {'entry': rates['close'][-1], 'sl': None, 'tp': None}
        elif hhhc:
            return 'sell', {'entry': rates['close'][-1], 'sl': None, 'tp': None}
        return None, None

    def special_104_calculate_trade_decision(self, rates):
        """
        Special Strategy 104: Based on Bollinger Bands.

        Returns:
            Tuple: (decision, trade_data)
        """
        bb_period = 20  # Assuming default BB period
        if len(rates) < bb_period:
            return None, None

        close_prices = rates['close']
        upper_band, middle_band, lower_band = talib.BBANDS(
            close_prices,
            timeperiod=bb_period,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )

        if (close_prices[-1] > upper_band[-1] and close_prices[-2] > upper_band[-2] and close_prices[-3] < upper_band[-3]):
            return 'buy', {'entry': close_prices[-1], 'sl': lower_band[-1], 'tp': upper_band[-1]}
        elif (close_prices[-1] < lower_band[-1] and close_prices[-2] < lower_band[-2] and close_prices[-3] > lower_band[-3]):
            return 'sell', {'entry': close_prices[-1], 'sl': upper_band[-1], 'tp': lower_band[-1]}
        return None, None

    # Placeholder for other special strategies
    def special_105_calculate_trade_decision(self, rates):
        return None, None

    def special_106_calculate_trade_decision(self, rates):
        return None, None

    def special_107_calculate_trade_decision(self, rates):
        return None, None