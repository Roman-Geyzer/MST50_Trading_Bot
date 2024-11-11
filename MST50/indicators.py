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
    get_rsi_indicator: Return the RSI indicator instance for calculation.
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
        self.indicator_name = self.strategy_config['indicator']
        self.indicator_instance = self.initialize_indicator()

    def initialize_indicator(self):
        """
        Initialize the correct indicator instance based on the strategy configuration.

        Returns:
            Indicator: An instance of the corresponding indicator class.
        """
        if self.indicator_name in self.indicator_mapping:
            indicator_class = self.indicator_mapping[self.indicator_name]
            return indicator_class(self.params)
        else:
            return NoIndicator(self.params)  # Fallback to no indicator

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

    def get_rsi_indicator(self):
        """
        Return the RSI indicator instance for calculation.
        """
        return self.indicator_instance if isinstance(self.indicator_instance, RSIIndicator) else None

class Indicator:
    def __init__(self, params):
        """
        Initialize the base Indicator class with common parameters.

        Parameters:
            params (dict): A dictionary of indicator parameters (common across all indicators).
        """
        self.params = params
        self.trade_decision_method = None  # To be set by subclasses

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

class NoIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the No Indicator class with the given parameters, inherited from the Indicator superclass.

        Parameters:
            params (dict): A dictionary of parameters for the No Indicator.
        """
        super().__init__(params)
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
        Initialize the Bollinger Bands Indicator with the given parameters, inherited from the Indicator superclass.

        Parameters:
            params (dict): A dictionary of parameters for the BB indicator (e.g., period, deviation).
        """
        super().__init__(params)
        self.bb_period = safe_int_extract_from_dict(params, 'a', 20)  # Default period of 20 if not provided
        self.bb_deviation = safe_float_extract_from_dict(params, 'b', 2.0)  # Default deviation of 2.0 if not provided

        # Mapping for BB trade decision methods
        self.decision_methods = {
            'BB_With': self.calculate_bb_with,
            'BB_Return': self.calculate_bb_return,
            'BB_Over': self.calculate_bb_over,
        }

        # Store the specific trade decision method based on the parameters
        self.trade_decision_method = self.get_trade_decision_method()

    def calculate_indicator_rates(self, rates):
        """
        Calculate the Bollinger Bands (upper, middle, lower) using historical price data (OHLC).
        """
        close_prices = rates['close']

        if len(close_prices) < self.bb_period:
            return None, None, None

        # Use talib to calculate Bollinger Bands
        upper_band, middle_band, lower_band = talib.BBANDS(
            close_prices,
            timeperiod=self.bb_period,
            nbdevup=self.bb_deviation,
            nbdevdn=self.bb_deviation,
            matype=0  # Simple Moving Average
        )

        return upper_band, middle_band, lower_band

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator based on the parameters.

        Returns:
            function: The trade decision method.
        """
        indicator_type = self.params.get('type', 'BB_With')
        return self.decision_methods.get(indicator_type, self.calculate_bb_with)

    def calculate_bb_with(self, rates):
        """
        Determine trade decision based on BB_With strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        upper_band, middle_band, lower_band = self.calculate_indicator_rates(rates)
        if upper_band is None:
            return None, None

        current_close = rates['close'][-1]

        if current_close > upper_band[-1]:
            return 'buy', {'entry': current_close, 'sl': lower_band[-1], 'tp': upper_band[-1] + 10}
        elif current_close < lower_band[-1]:
            return 'sell', {'entry': current_close, 'sl': upper_band[-1], 'tp': lower_band[-1] - 10}
        return None, None

    def calculate_bb_return(self, rates):
        """
        Determine trade decision based on BB_Return strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        upper_band, middle_band, lower_band = self.calculate_indicator_rates(rates)
        if upper_band is None:
            return None, None

        close = rates['close']
        if close[-1] > lower_band[-1] and close[-2] < lower_band[-2]:
            return 'buy', {'entry': close[-1], 'sl': lower_band[-1], 'tp': upper_band[-1]}
        elif close[-1] < upper_band[-1] and close[-2] > upper_band[-2]:
            return 'sell', {'entry': close[-1], 'sl': upper_band[-1], 'tp': lower_band[-1]}
        return None, None

    def calculate_bb_over(self, rates):
        """
        Determine trade decision based on BB_Over strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        upper_band, middle_band, lower_band = self.calculate_indicator_rates(rates)
        if upper_band is None:
            return None, None

        current_close = rates['close'][-1]

        if current_close > middle_band[-1]:
            return 'buy', {'entry': current_close, 'sl': lower_band[-1], 'tp': upper_band[-1]}
        elif current_close < middle_band[-1]:
            return 'sell', {'entry': current_close, 'sl': upper_band[-1], 'tp': lower_band[-1]}
        return None, None

class MAIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Moving Average Indicator with the given parameters, inherited from the Indicator superclass.

        Parameters:
            params (dict): A dictionary of parameters for the MA indicator.
        """
        super().__init__(params)
        self.fast_ma_period = safe_int_extract_from_dict(params, 'a', 7)  # Fast MA period
        self.slow_ma_period = safe_int_extract_from_dict(params, 'b', 21)  # Slow MA period
        self.long_ma_period = safe_int_extract_from_dict(params, 'c', 50)  # Long MA period
        self.use_ema = safe_bool_extract_from_dict(params, 'd', False)  # Whether to use EMA instead of SMA

        # Store the specific trade decision method
        self.trade_decision_method = self.calculate_trade_decision

    def calculate_indicator_rates(self, rates):
        """
        Calculate Moving Averages (fast, slow, long) based on the historical price data.
        """
        close_prices = rates['close']

        # Calculate Fast, Slow, Long MA using talib
        if self.use_ema:
            self.fast_ma = talib.EMA(close_prices, timeperiod=self.fast_ma_period)
            self.slow_ma = talib.EMA(close_prices, timeperiod=self.slow_ma_period)
            self.long_ma = talib.EMA(close_prices, timeperiod=self.long_ma_period)
        else:
            self.fast_ma = talib.SMA(close_prices, timeperiod=self.fast_ma_period)
            self.slow_ma = talib.SMA(close_prices, timeperiod=self.slow_ma_period)
            self.long_ma = talib.SMA(close_prices, timeperiod=self.long_ma_period)

    def calculate_trade_decision(self, rates):
        """
        Determine trade decision based on Moving Average strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)

        if np.isnan(self.fast_ma[-1]) or np.isnan(self.slow_ma[-1]) or np.isnan(self.long_ma[-1]):
            return None, None

        current_close = rates['close'][-1]

        # Example trade decision logic based on MA alignment
        if self.fast_ma[-1] > self.slow_ma[-1] > self.long_ma[-1]:
            return 'buy', {'entry': current_close, 'sl': self.slow_ma[-1], 'tp': current_close + 10}
        elif self.fast_ma[-1] < self.slow_ma[-1] < self.long_ma[-1]:
            return 'sell', {'entry': current_close, 'sl': self.slow_ma[-1], 'tp': current_close - 10}
        return None, None

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

    def calculate_gr_ratio(self, rates):
        """
        Calculate the Green/Red (GR) ratio over the given period.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            float: The Green/Red ratio.
        """
        if len(rates) < self.ratio_candles_count:
            return None

        # Get the last N candles
        recent_rates = rates[-self.ratio_candles_count:]

        # Calculate green and red candles
        up_moves = recent_rates['close'] > recent_rates['open']
        down_moves = recent_rates['close'] < recent_rates['open']

        green_count = np.sum(up_moves)
        red_count = np.sum(down_moves)

        if red_count == 0:
            return float('inf')  # Infinite ratio (only green candles)

        gr_ratio = green_count / red_count

        return gr_ratio

    def gr_check_is_buy(self, gr_ratio):
        """
        Check if the GR ratio signals a buy entry.

        Parameters:
            gr_ratio (float): The current GR ratio.

        Returns:
            bool: True if it's a buy signal, False otherwise.
        """
        return gr_ratio > self.buy_enter_limit

    def gr_check_is_sell(self, gr_ratio):
        """
        Check if the GR ratio signals a sell entry.

        Parameters:
            gr_ratio (float): The current GR ratio.

        Returns:
            bool: True if it's a sell signal, False otherwise.
        """
        return (1 / gr_ratio) > self.sell_enter_limit

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on the GR ratio.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        gr_ratio = self.calculate_gr_ratio(rates)
        if gr_ratio is None:
            return None, None

        current_close = rates['close'][-1]
        current_low = rates['low'][-1]
        current_high = rates['high'][-1]

        # If GR ratio signals a buy
        if self.gr_check_is_buy(gr_ratio):
            return 'buy', {'entry': current_close, 'sl': current_low - 10, 'tp': current_close + 20}

        # If GR ratio signals a sell
        if self.gr_check_is_sell(gr_ratio):
            return 'sell', {'entry': current_close, 'sl': current_high + 10, 'tp': current_close - 20}

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

    def calculate_indicator_rates(self, rates):
        """
        Calculate the RSI based on the given OHLC data using talib.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            np.ndarray: Calculated RSI values.
        """
        close_prices = rates['close']
        if len(close_prices) < self.rsi_period:
            return None

        rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
        return rsi

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
        self.rsi_values = self.calculate_indicator_rates(rates)
        if self.rsi_values is None:
            return None, None

        if self.rsi_divergence_check_is_buy(rates):
            return 'buy', {'entry': rates['close'][-1], 'sl': rates['low'][-1] - 10, 'tp': rates['close'][-1] + 20}
        elif self.rsi_divergence_check_is_sell(rates):
            return 'sell', {'entry': rates['close'][-1], 'sl': rates['high'][-1] + 10, 'tp': rates['close'][-1] - 20}
        return None, None

    def calculate_rsi_div_hidden(self, rates):
        """
        Determine trade decision based on hidden RSI divergence strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.rsi_values = self.calculate_indicator_rates(rates)
        if self.rsi_values is None:
            return None, None

        if self.rsi_hidden_divergence_check_is_buy(rates):
            return 'buy', {'entry': rates['close'][-1], 'sl': rates['low'][-1] - 10, 'tp': rates['close'][-1] + 20}
        elif self.rsi_hidden_divergence_check_is_sell(rates):
            return 'sell', {'entry': rates['close'][-1], 'sl': rates['high'][-1] + 10, 'tp': rates['close'][-1] - 20}
        return None, None

    def calculate_rsi_over(self, rates):
        """
        Determine trade decision based on RSI overbought/oversold levels.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.rsi_values = self.calculate_indicator_rates(rates)
        if self.rsi_values is None:
            return None, None

        if self.rsi_oversold_check_is_buy():
            return 'buy', {'entry': rates['close'][-1], 'sl': rates['low'][-1] - 10, 'tp': rates['close'][-1] + 20}
        elif self.rsi_overbought_check_is_sell():
            return 'sell', {'entry': rates['close'][-1], 'sl': rates['high'][-1] + 10, 'tp': rates['close'][-1] - 20}
        return None, None

    def calculate_rsi_with(self, rates):
        """
        Determine trade decision based on RSI trending with overbought/oversold levels.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.rsi_values = self.calculate_indicator_rates(rates)
        if self.rsi_values is None:
            return None, None

        if self.rsi_with_trend_check_is_buy():
            return 'buy', {'entry': rates['close'][-1], 'sl': rates['low'][-1] - 10, 'tp': rates['close'][-1] + 20}
        elif self.rsi_with_trend_check_is_sell():
            return 'sell', {'entry': rates['close'][-1], 'sl': rates['high'][-1] + 10, 'tp': rates['close'][-1] - 20}
        return None, None

    def rsi_divergence_check_is_buy(self, rates):
        """
        Check for a bullish RSI divergence.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if a bullish divergence is detected, False otherwise.
        """
        idx = -3  # Current index
        if not self.is_local_rsi_min(idx):
            return False

        for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
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
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if a bearish divergence is detected, False otherwise.
        """
        idx = -3  # Current index
        if not self.is_local_rsi_max(idx):
            return False

        for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
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
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if a hidden bullish divergence is detected, False otherwise.
        """
        idx = -3
        if not self.is_local_rsi_min(idx):
            return False
        if self.rsi_values[idx] < 50 - self.rsi_over_extended:
            for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
                if self.rsi_values[i] > self.rsi_values[idx]:
                    if self.is_local_rsi_min(i):
                        if rates['close'][i] < rates['close'][idx]:
                            return True
        return False

    def rsi_hidden_divergence_check_is_sell(self, rates):
        """
        Check for a hidden bearish RSI divergence.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if a hidden bearish divergence is detected, False otherwise.
        """
        idx = -3
        if not self.is_local_rsi_max(idx):
            return False
        if self.rsi_values[idx] > 50 + self.rsi_over_extended:
            for i in range(idx - 3, idx - self.rsi_div_lookback_period, -1):
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
        return (
            self.rsi_values[idx] > self.rsi_values[idx + 1] and self.rsi_values[idx] > self.rsi_values[idx + 2] and
            self.rsi_values[idx] > self.rsi_values[idx - 1] and self.rsi_values[idx] > self.rsi_values[idx - 2]
        )

    def is_local_rsi_min(self, idx):
        """
        Check if the RSI value at a given index is a local minimum.

        Parameters:
            idx (int): Index to check for local min.

        Returns:
            bool: True if the RSI is a local min, False otherwise.
        """
        return (
            self.rsi_values[idx] < self.rsi_values[idx + 1] and self.rsi_values[idx] < self.rsi_values[idx + 2] and
            self.rsi_values[idx] < self.rsi_values[idx - 1] and self.rsi_values[idx] < self.rsi_values[idx - 2]
        )

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
    def __init__(self, params):
        """
        Initialize the Range Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for SR, Breakout, and Fakeout calculations.
        """
        super().__init__(params)
        self.period_for_sr = safe_int_extract_from_dict(params, 'a', 100)
        self.slack_for_sr_perc_div = safe_float_extract_from_dict(params, 'c', 10.0)
        self.perc_rejection_multiplier = safe_float_extract_from_dict(params, 'd', 1.0)
        self.max_distance_from_sr_perc = safe_float_extract_from_dict(params, 'e', 2.0)
        self.min_height_of_sr_distance = safe_float_extract_from_dict(params, 'f', 3.0)
        self.max_height_of_sr_distance = safe_float_extract_from_dict(params, 'g', 200.0)
        self.bars_from_fakeout = safe_int_extract_from_dict(params, 'h', 2)
        self.bars_before_fakeout = safe_int_extract_from_dict(params, 'i', 2)
        self.fakeout_perc_slack = safe_float_extract_from_dict(params, 'j', 0.5)
        self.slack_for_breakout_perc = safe_float_extract_from_dict(params, 'k', 0.1)

        # Initialize SR levels
        self.upper_sr = 0.0
        self.lower_sr = 0.0
        self.prev_upper_sr_level = 0.0
        self.prev_lower_sr_level = 0.0

        # Determine the strategy type
        self.strategy_type = self.params.get('type', 'SR')

        # Mapping for trade decision methods
        self.decision_methods = {
            'SR': self.sr_trade_decision,
            'Breakout': self.breakout_trade_decision,
            'Fakeout': self.fakeout_trade_decision
        }

        # Store the specific trade decision method based on the parameters
        self.trade_decision_method = self.get_trade_decision_method()

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator based on the parameters.

        Returns:
            function: The trade decision method.
        """
        return self.decision_methods.get(self.strategy_type, self.sr_trade_decision)

    def calculate_sr_levels(self, rates):
        """
        Calculate the Support and Resistance (SR) levels for the given rates.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
        """
        if len(rates) < self.period_for_sr:
            return

        recent_rates = rates[-self.period_for_sr:]

        highest_high = np.max(recent_rates['high'])
        lowest_low = np.min(recent_rates['low'])

        # Adjust levels based on slack percentage
        price_range = highest_high - lowest_low
        slack = price_range / self.slack_for_sr_perc_div

        self.upper_sr = highest_high + slack
        self.lower_sr = lowest_low - slack

        # Store previous SR levels for Breakout and Fakeout checks
        self.prev_upper_sr_level = self.upper_sr
        self.prev_lower_sr_level = self.lower_sr

    def sr_trade_decision(self, rates):
        """
        Make a trade decision based on SR strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_sr_levels(rates)
        current_open = rates['open'][-1]
        current_close = rates['close'][-1]

        distance_from_upper = self.upper_sr - current_open
        distance_from_lower = current_open - self.lower_sr
        price_range = current_open

        if distance_from_lower < (self.max_distance_from_sr_perc / 100) * price_range:
            # Buy signal
            return 'buy', {'entry': current_close, 'sl': self.lower_sr - 10, 'tp': self.upper_sr}
        elif distance_from_upper < (self.max_distance_from_sr_perc / 100) * price_range:
            # Sell signal
            return 'sell', {'entry': current_close, 'sl': self.upper_sr + 10, 'tp': self.lower_sr}
        return None, None

    def breakout_trade_decision(self, rates):
        """
        Make a trade decision based on Breakout strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_sr_levels(rates)
        current_close = rates['close'][-1]
        previous_close = rates['close'][-2]

        if previous_close > self.prev_upper_sr_level + (self.slack_for_breakout_perc / 100) * previous_close:
            # Buy signal
            return 'buy', {'entry': current_close, 'sl': self.prev_lower_sr_level - 10, 'tp': current_close + 20}
        elif previous_close < self.prev_lower_sr_level - (self.slack_for_breakout_perc / 100) * previous_close:
            # Sell signal
            return 'sell', {'entry': current_close, 'sl': self.prev_upper_sr_level + 10, 'tp': current_close - 20}
        return None, None

    def fakeout_trade_decision(self, rates):
        """
        Make a trade decision based on Fakeout strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_sr_levels(rates)
        total_bars_needed = self.bars_from_fakeout + self.bars_before_fakeout
        if len(rates) < total_bars_needed:
            return None, None

        recent_rates = rates[-total_bars_needed:]

        lows = recent_rates['low']
        highs = recent_rates['high']

        lowest_in_fakeout = np.min(lows[-self.bars_from_fakeout:])
        previous_lowest = np.min(lows[:-self.bars_from_fakeout])

        highest_in_fakeout = np.max(highs[-self.bars_from_fakeout:])
        previous_highest = np.max(highs[:-self.bars_from_fakeout])

        if (lowest_in_fakeout > self.lower_sr - (self.fakeout_perc_slack / 100) * lowest_in_fakeout) and (previous_lowest < self.lower_sr):
            # Buy signal
            return 'buy', {'entry': rates['close'][-1], 'sl': self.lower_sr - 10, 'tp': self.upper_sr}
        elif (highest_in_fakeout < self.upper_sr + (self.fakeout_perc_slack / 100) * highest_in_fakeout) and (previous_highest > self.upper_sr):
            # Sell signal
            return 'sell', {'entry': rates['close'][-1], 'sl': self.upper_sr + 10, 'tp': self.lower_sr}
        return None, None

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on SR, Breakout, or Fakeout conditions.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

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