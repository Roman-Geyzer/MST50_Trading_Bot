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
Main Functions:
    initialize_indicator: Initialize the correct indicator instance based on the strategy configuration.
    make_trade_decision: Make a trade decision based on the selected indicator class.
    get_rsi_indicator: Return the RSI indicator instance for calculation.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from .constants import (TIMEFRAME_MT5_MAPPING, TRADE_DIRECTION, DEAL_TYPE, TRADE_TYPE, TRADE_DECISION)
from .utils import (safe_int_extract_from_dict, safe_float_extract_from_dict, safe_str_extract_from_dict, safe_bool_extract_from_dict,
                    print_hashtaged_msg)
from .candles import Candle


# TODO: how do I use this class?
# TODO: Do I need this class?
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

        # First layer mapping to indicator classes
        self.indicator_mapping = {
            '0': NoIndicator,  # No indicator
            'SR': RangeIndicator,
            'Breakout': RangeIndicator,
            'Fakeout': RangeIndicator,
            'Trend': TrendIndicator,
            'TrendBreak': TrendIndicator,
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
            rates  : Historical price data (OHLC).
        
        Returns:
            Tuple: (decision, trade_data) - where decision is a string (e.g., 'buy', 'sell', None)
                   and trade_data is a dictionary with trade parameters (entry price, stop loss, take profit).
        """
        # Check if the indicator instance exists
        if self.indicator_instance:
            #TODO: move this check to the indicator class when the instrance is initiated
            if len(rates) >= self.params.get('a'):
                return self.indicator_instance.claculuate_and_make_make_trade_decision(rates)
        else:
            return None, None

    def get_rsi_indicator(self):
        """
        Return the RSI indicator instance for calculation.
        """
        return self.rsi_indicator





# TODO: GPT code - validate


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
        A placeholder method to calculate indicator rates. This should be overridden by subclasses if necessary.
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
            rates  : Historical price data (OHLC).

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



#TODO: Implement return trade values for the following classe
#TODO: Update the following classes to recieve the rates as a dataframe and not a list of dictionaries
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class BBIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Bollinger Bands Indicator with the given parameters, inherited from the Indicator superclass.

        Parameters:
            params (dict): A dictionary of parameters for the BB indicator (e.g., period, deviation).
        """
        super().__init__(params)
        self.bb_period = int(params.get('a', 20))  # Default period of 20 if not provided
        self.bb_deviation = params.get('b', 2.0)  # Default deviation of 2.0 if not provided
        self.upper_band = []
        self.middle_band = []
        self.lower_band = []

        # Mapping for BB trade decision methods (specific to BBIndicator)
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
        close_prices = [rate['close'] for rate in rates]

        if len(close_prices) >= self.bb_period:
            for i in range(self.bb_period, len(close_prices)):
                window = close_prices[i - self.bb_period:i]
                middle_band = sum(window) / self.bb_period
                std_dev = (sum([(x - middle_band) ** 2 for x in window]) / self.bb_period) ** 0.5

                upper_band = middle_band + self.bb_deviation * std_dev
                lower_band = middle_band - self.bb_deviation * std_dev

                self.upper_band.append(upper_band)
                self.middle_band.append(middle_band)
                self.lower_band.append(lower_band)

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator based on the parameters.

        Returns:
            function: The trade decision method.
        """
        return self.decision_methods.get(self.params.get('type', 'BB_With'))

    def calculate_bb_with(self, rates):
        """
        Determine trade decision based on BB_With strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if rates[-1]['close'] > self.upper_band[-1]:
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_band[-1], 'tp': self.upper_band[-1] + 10}
        elif rates[-1]['close'] < self.lower_band[-1]:
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_band[-1], 'tp': self.lower_band[-1] - 10}
        return None, None

    def calculate_bb_return(self, rates):
        """
        Determine trade decision based on BB_Return strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if rates[-1]['close'] > self.lower_band[-1] and rates[-2]['close'] < self.lower_band[-2]:
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_band[-1], 'tp': self.upper_band[-1]}
        elif rates[-1]['close'] < self.upper_band[-1] and rates[-2]['close'] > self.upper_band[-2]:
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_band[-1], 'tp': self.lower_band[-1]}
        return None, None

    def calculate_bb_over(self, rates):
        """
        Determine trade decision based on BB_Over strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if rates[-1]['close'] > self.middle_band[-1]:
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_band[-1], 'tp': self.upper_band[-1]}
        elif rates[-1]['close'] < self.middle_band[-1]:
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_band[-1], 'tp': self.lower_band[-1]}
        return None, None



#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
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


        # Initialize MA buffers
        self.fast_ma = []
        self.slow_ma = []
        self.long_ma = []

        # Store the specific trade decision method
        self.trade_decision_method = self.get_trade_decision_method()

    def calculate_indicator_rates(self, rates):
        """
        Calculate Moving Averages (fast, slow, long) based on the historical price data.
        """
        close_prices = [rate['close'] for rate in rates]

        # Calculate Fast, Slow, Long MA
        self.fast_ma = self.calculate_ma(close_prices, self.fast_ma_period)
        self.slow_ma = self.calculate_ma(close_prices, self.slow_ma_period)
        self.long_ma = self.calculate_ma(close_prices, self.long_ma_period)

    def calculate_ma(self, prices, period):
        """
        Helper function to calculate simple moving average (or EMA if use_ema is True).
        """
        if len(prices) < period:
            return []

        ma = []
        if self.use_ema:
            # Exponential Moving Average calculation (simplified)
            ema = prices[0]
            smoothing = 2 / (period + 1)
            for price in prices[1:]:
                ema = (price - ema) * smoothing + ema
                ma.append(ema)
        else:
            # Simple Moving Average calculation
            for i in range(period, len(prices)):
                ma.append(sum(prices[i - period:i]) / period)

        return ma

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for the MA strategy.

        Returns:
            function: The trade decision method.
        """
        return self.calculate_trade_decision

    def calculate_trade_decision(self, rates):
        """
        Determine trade decision based on Moving Average strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)

        if len(self.fast_ma) == 0 or len(self.slow_ma) == 0 or len(self.long_ma) == 0:
            return None, None

        # Example trade decision logic based on MA alignment
        if self.fast_ma[-1] > self.slow_ma[-1] > self.long_ma[-1]:
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.slow_ma[-1], 'tp': self.fast_ma[-1] + 10}
        elif self.fast_ma[-1] < self.slow_ma[-1] < self.long_ma[-1]:
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.slow_ma[-1], 'tp': self.fast_ma[-1] - 10}
        return None, None



#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class DoubleIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Double Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for Double pattern detection.
        """
        super().__init__(params)
        self.period_for_double = int(params.get('a') or 100)  # Lookback period for double
        self.slack_perc = float(params.get('b') or 0.1)  # Slack for double percentage
        self.wait_between_candles = int(params.get('c', 6))  # Min candles between double tops/bottoms
        self.max_bars_from_double = int(params.get('d', 5))  # Max bars after double to trigger a trade
        self.max_distance_perc = float(params.get('e', 3))  # Max distance from double percentage
        self.min_distance_perc = float(params.get('f', 0.1))  # Min distance from double percentage

        # Initialize state variables
        self.double_up = False
        self.double_down = False
        self.double_up_price = 0.0
        self.double_down_price = 0.0
        self.first_touch_value = 0.0
        self.second_touch_value = 0.0

        # Set the trade decision method
        self.trade_decision_method = self.claculuate_and_make_make_trade_decision

    @staticmethod
    def find_extreme(rates, period: int, mode: str):
        """
        Helper function to find the minimum or maximum in the given period.

        Parameters:
            rates  : Historical rates.
            period (int): Lookback period.
            mode (str): Either 'low' for lowest points or 'high' for highest points.

        Returns:
            dict: The rate dict with the min or max value.
        """
        if mode == 'low':
            temp = min(rates[-period:], key=lambda x: x['low'])
        elif mode == 'high':
            temp = max(rates[-period:], key=lambda x: x['high'])
        else:
            print_hashtaged_msg(1, "Should not be here!!!!!" , f"Invalid mode: {mode}")
            return None
        return temp

    def calculate_double_up(self, rates):
        """
        Identify double bottom (up) pattern and trigger a potential buy signal.
        """
        temp1 = self.find_extreme(rates, self.period_for_double, 'low')
        temp2 = self.find_extreme(rates, self.period_for_double - 1, 'low')

        if not temp1 or not temp2:
            return

        # Calculate the difference in candle indices
        candle_diff = abs(temp1['index'] - temp2['index'])
        if candle_diff < self.wait_between_candles:
            return

        self.first_touch_value = temp1['low']
        self.second_touch_value = temp2['low']

        slack_threshold = self.slack_perc * rates[-1]['close'] / 100
        if abs(self.first_touch_value - self.second_touch_value) > slack_threshold:
            return

        if rates[-1]['open'] < self.second_touch_value:
            return

        self.double_up = True
        self.double_up_price = self.second_touch_value

    def calculate_double_down(self, rates):
        """
        Identify double top (down) pattern and trigger a potential sell signal.
        """
        temp1 = self.find_extreme(rates, self.period_for_double, 'high')
        temp2 = self.find_extreme(rates, self.period_for_double - 1, 'high')

        if not temp1 or not temp2:
            return

        # Calculate the difference in candle indices
        candle_diff = abs(temp1['index'] - temp2['index'])
        if candle_diff < self.wait_between_candles:
            return

        self.first_touch_value = temp1['high']
        self.second_touch_value = temp2['high']

        slack_threshold = self.slack_perc * rates[-1]['close'] / 100
        if abs(self.first_touch_value - self.second_touch_value) > slack_threshold:
            return

        if rates[-1]['open'] > self.second_touch_value:
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
            rates : Historical price data (OHLC) with 'index' key.

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        self.calculate_double_up(rates)
        self.calculate_double_down(rates)

        if self.double_check_is_buy():
            trade_data = {
                'entry': rates[-1]['close'],
                'sl': self.double_up_price - 10,  # Adjust SL as needed
                'tp': rates[-1]['close'] + 20     # Adjust TP as needed
            }
            return 'buy', trade_data

        if self.double_check_is_sell():
            trade_data = {
                'entry': rates[-1]['close'],
                'sl': self.double_down_price + 10,  # Adjust SL as needed
                'tp': rates[-1]['close'] - 20      # Adjust TP as needed
            }
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

    def get_trade_decision_method(self):
        """
        Return the trade decision method for this indicator.
        
        Returns:
            function: The trade decision method.
        """
        return self.trade_decision_method



#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class GRIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the GR Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for GR Ratio calculation.
        """
        super().__init__(params)
        self.ratio_candles_count = int(params.get('a', 100))  # Number of candles to calculate ratio
        self.buy_enter_limit = float(params.get('b', 1.35))  # Buy entry limit for GR ratio
        self.sell_enter_limit = float(params.get('c', 1.35))  # Sell entry limit for GR ratio
        self.buy_exit_limit = float(params.get('d', 1))  # Buy exit limit for GR ratio
        self.sell_exit_limit = float(params.get('e', 1))  # Sell exit limit for GR ratio
        self.gr_ratios = []  # Store GR ratios for historical calculations

    def calculate_gr_ratio(self, rates):
        """
        Calculate the Green/Red (GR) ratio over the given period.

        Parameters:
            rates : Historical price data (OHLC).
        
        Returns:
            float: The Green/Red ratio.
        """
        green_count = 0
        red_count = 0

        for i in range(1, self.ratio_candles_count + 1):
            if Candle.candle_color(rates[-i]) == 1:  # Green candle
                green_count += 1
            elif Candle.candle_color(rates[-i]) == -1:  # Red candle
                red_count += 1

        # Avoid division by zero by returning 0.0 if no red candles
        if red_count == 0:
            return float('inf')  # Infinite ratio (only green candles)
        
        gr_ratio = green_count / red_count
        self.gr_ratios.append(gr_ratio)  # Store ratio for historical analysis
        
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

    def manage_gr_exit(self, position_type, gr_ratio):
        """
        Check exit conditions based on the GR ratio.

        Parameters:
            position_type (str): The current position type ('buy' or 'sell').
            gr_ratio (float): The current GR ratio.
        
        Returns:
            bool: True if an exit is needed, False otherwise.
        """
        if position_type == 'buy' and gr_ratio < self.buy_exit_limit:
            return True
        if position_type == 'sell' and (1 / gr_ratio) < self.sell_exit_limit:
            return True
        return False

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on the GR ratio.

        Parameters:
            rates : Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        gr_ratio = self.calculate_gr_ratio(rates)
        
        # If GR ratio signals a buy
        if self.gr_check_is_buy(gr_ratio):
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}

        # If GR ratio signals a sell
        if self.gr_check_is_sell(gr_ratio):
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}

        return None, None

    def manage_exit_gr(self, position, rates):
        """
        Manage exit conditions based on GR ratio.

        Parameters:
            position (dict): Current open position data.
            rates : Historical price data (OHLC).
        
        Returns:
            bool: True if position should be closed, False otherwise.
        """
        gr_ratio = self.calculate_gr_ratio(rates)
        position_type = 'buy' if position['type'] == 'buy' else 'sell'

        return self.manage_gr_exit(position_type, gr_ratio)



#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class RSIIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the RSI Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for RSI calculation.
        """
        super().__init__(params)
        self.rsi_period = safe_int_extract_from_dict(params, 'a', 14)  # RSI period
        self.rsi_over_extended = safe_int_extract_from_dict(params, 'b', 20)
        self.rsi_div_lookback_period = safe_int_extract_from_dict(params, 'c', 50)
        self.rsi_values = []  # Store RSI values for historical calculations

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
        Calculate the RSI based on the given OHLC data (using the standard RSI calculation).

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
           Calculated RSI values.
        """
        # Extract close prices from rates
        close_prices = [bar['close'] for bar in rates]
        if len(close_prices) < self.rsi_period + 1:
            return []

        # Calculate price changes
        deltas = np.diff(close_prices)
        seed = deltas[:self.rsi_period]
        up = seed[seed >= 0].sum() / self.rsi_period
        down = -seed[seed < 0].sum() / self.rsi_period
        rs = up / down if down != 0 else 0
        rsi = [100.0 - (100.0 / (1.0 + rs))]

        # Calculate RSI for the rest of the data
        for delta in deltas[self.rsi_period:]:
            if delta > 0:
                up_val = delta
                down_val = 0
            else:
                up_val = 0
                down_val = -delta

            up = (up * (self.rsi_period - 1) + up_val) / self.rsi_period
            down = (down * (self.rsi_period - 1) + down_val) / self.rsi_period

            rs = up / down if down != 0 else 0
            rsi_val = 100.0 - (100.0 / (1.0 + rs))
            rsi.append(rsi_val)

        return rsi

    def get_trade_decision_method(self):
        """
        Return the appropriate trade decision method for this indicator.

        Returns:
            function: The trade decision method.
        """
        return self.decision_methods.get(self.params.get('type', 'RSI_Div'))

    def calculate_rsi_div(self, rates):
        """
        Determine trade decision based on RSI divergence strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if not self.rsi_values:
            return None, None
        if self.rsi_divergence_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}
        elif self.rsi_divergence_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}
        return None, None

    def calculate_rsi_div_hidden(self, rates):
        """
        Determine trade decision based on hidden RSI divergence strategy.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if not self.rsi_values:
            return None, None
        if self.rsi_hidden_divergence_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}
        elif self.rsi_hidden_divergence_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}
        return None, None

    def calculate_rsi_over(self, rates):
        """
        Determine trade decision based on RSI overbought/oversold levels.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if not self.rsi_values:
            return None, None
        if self.rsi_oversold_check_is_buy():
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}
        elif self.rsi_overbought_check_is_sell():
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}
        return None, None

    def calculate_rsi_with(self, rates):
        """
        Determine trade decision based on RSI trending with overbought/oversold levels.

        Returns:
            Tuple: (decision, trade_data)
        """
        self.calculate_indicator_rates(rates)
        if not self.rsi_values:
            return None, None
        if self.rsi_with_trend_check_is_buy():
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}
        elif self.rsi_with_trend_check_is_sell():
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}
        return None, None

    def rsi_divergence_check_is_buy(self, rates):
        """
        Check for a bullish RSI divergence.

        Parameters:
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if a bullish divergence is detected, False otherwise.
        """
        if not self.is_local_rsi_min(3):
            return False
        for i in range(6, self.rsi_div_lookback_period):
            if self.rsi_values[i] < 50 - self.rsi_over_extended:
                if self.rsi_values[i] < self.rsi_values[3]:
                    if self.is_local_rsi_min(i):
                        if rates[i]['close'] > rates[3]['close']:
                            return True
        return False

    def rsi_divergence_check_is_sell(self, rates):
        """
        Check for a bearish RSI divergence.

        Parameters:
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if a bearish divergence is detected, False otherwise.
        """
        if not self.is_local_rsi_max(3):
            return False
        for i in range(6, self.rsi_div_lookback_period):
            if self.rsi_values[i] > 50 + self.rsi_over_extended:
                if self.rsi_values[i] > self.rsi_values[3]:
                    if self.is_local_rsi_max(i):
                        if rates[i]['close'] < rates[3]['close']:
                            return True
        return False

    def rsi_hidden_divergence_check_is_buy(self, rates):
        """
        Check for a hidden bullish RSI divergence.

        Parameters:
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if a hidden bullish divergence is detected, False otherwise.
        """
        if not self.is_local_rsi_min(3):
            return False
        if self.rsi_values[3] < 50 - self.rsi_over_extended:
            for i in range(6, self.rsi_div_lookback_period):
                if self.rsi_values[i] > self.rsi_values[3]:
                    if self.is_local_rsi_min(i):
                        if rates[i]['close'] < rates[3]['close']:
                            return True
        return False

    def rsi_hidden_divergence_check_is_sell(self, rates):
        """
        Check for a hidden bearish RSI divergence.

        Parameters:
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if a hidden bearish divergence is detected, False otherwise.
        """
        if not self.is_local_rsi_max(3):
            return False
        if self.rsi_values[3] > 50 + self.rsi_over_extended:
            for i in range(6, self.rsi_div_lookback_period):
                if self.rsi_values[i] < self.rsi_values[3]:
                    if self.is_local_rsi_max(i):
                        if rates[i]['close'] > rates[3]['close']:
                            return True
        return False

    def rsi_overbought_check_is_sell(self):
        """
        Check if the RSI indicates overbought conditions (sell signal).
        """

        return self.rsi_values[3] > 50 + self.rsi_over_extended and self.is_local_rsi_max(3)

    def rsi_oversold_check_is_buy(self):
        """
        Check if the RSI indicates oversold conditions (buy signal).
        """
        return self.rsi_values[3] < 50 - self.rsi_over_extended and self.is_local_rsi_min(3)

    def rsi_with_trend_check_is_buy(self):
        """
        Check if RSI indicates a buy signal when trending with oversold conditions.
        """
        return self.rsi_values[1] > 50 + self.rsi_over_extended

    def rsi_with_trend_check_is_sell(self):
        """
        Check if RSI indicates a sell signal when trending with overbought conditions.
        """
        return self.rsi_values[1] < 50 - self.rsi_over_extended

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



#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class KAMAIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the KAMA Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for KAMA calculation.
        """
        super().__init__(params)
        self.er_candles_count = int(params.get('a', 20))  # Number of candles for calculating ER
        self.sc1 = 0.602151  # Fast SC constant
        self.sc2 = 0.064516  # Slow SC constant
        self.kama = 0
        self.prev_kama = 0

    def calculate_er_for_kama(self, rates, start_idx):
        """
        Calculate Efficiency Ratio (ER) for the KAMA calculation.

        Parameters:
            rates  : Historical price data (OHLC).
            start_idx (int): Index from which to start ER calculation.
        
        Returns:
            float: Efficiency Ratio.
        """
        change = abs(rates[start_idx]['close'] - rates[start_idx + self.er_candles_count]['close'])
        derivative = sum(abs(rates[i]['close'] - rates[i + 1]['close']) for i in range(start_idx, start_idx + self.er_candles_count))
        if derivative == 0:
            return 0.0
        return change / derivative

    def initiate_kama(self, rates):
        """
        Initialize the KAMA values for the indicator.

        Parameters:
            rates  : Historical price data (OHLC).
        """
        self.prev_kama = rates[-self.er_candles_count]['close']  # Initialize prev_kama with close price from history
        for i in range(self.er_candles_count - 1, 1, -1):
            er = self.calculate_er_for_kama(rates, i)
            sc = (er * self.sc1 + self.sc2) ** 2  # Smoothing constant
            self.kama = self.prev_kama + sc * (rates[i]['close'] - self.prev_kama)
            self.prev_kama = self.kama

    def calculate_er_ratio(self, rates):
        """
        Calculate the Efficiency Ratio (ER) based on the given candles.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            float: The Efficiency Ratio.
        """
        change = abs(rates[-1]['close'] - rates[-(self.candles_count + 1)]['close'])
        derivative = sum(abs(rates[-i]['close'] - rates[-(i+1)]['close']) for i in range(1, self.candles_count + 1))
        if derivative == 0:
            return 0.0  # Avoid division by zero
        return change / derivative

    def update_kama(self, rates):
        """
        Update the KAMA value with the latest price.

        Parameters:
            rates  : Historical price data (OHLC).
        """
        er = self.calculate_er_for_kama(rates, -1)  # Calculate ER for the latest data
        sc = (er * self.sc1 + self.sc2) ** 2  # Smoothing constant
        self.prev_kama = self.kama
        self.kama = self.prev_kama + sc * (rates[-1]['close'] - self.prev_kama)

    def kama_check_is_buy(self, rates):
        """
        Check if the KAMA indicator signals a buy.

        Parameters:
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if KAMA indicates a buy, False otherwise.
        """
        self.update_kama(rates)
        return rates[-1]['close'] > self.kama and rates[-2]['close'] < self.prev_kama

    def kama_check_is_sell(self, rates):
        """
        Check if the KAMA indicator signals a sell.

        Parameters:
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if KAMA indicates a sell, False otherwise.
        """
        self.update_kama(rates)
        return rates[-1]['close'] < self.kama and rates[-2]['close'] > self.prev_kama

    def manage_kama_exit(self, position, rates):
        """
        Manage exits based on KAMA signals.

        Parameters:
            position (dict): The current open position.
            rates  : Historical price data (OHLC).
        
        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        if position['type'] == 'buy' and self.kama_check_is_sell(rates):
            return True  # Exit buy position
        if position['type'] == 'sell' and self.kama_check_is_buy(rates):
            return True  # Exit sell position
        return False

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on KAMA signals.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        if self.kama_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}
        if self.kama_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}
        return None, None



#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class RangeIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Range Indicator with strategy parameters.
        
        Parameters:
            params (dict): Contains specific parameters for SR, Breakout, and Fakeout calculations.
        """
        super().__init__(params)
        self.period_for_sr = int(params.get('a') or 100)
        self.slack_for_sr_perc_div = float(params.get('c') or 10)
        self.perc_rejection_multiplier = float(params.get('d') or 1)
        self.max_distance_from_sr_perc = float(params.get('e') or 2)
        self.min_height_of_sr_distance = float(params.get('f') or 3)
        self.max_height_of_sr_distance = float(params.get('g') or 200)
        self.bars_from_fakeout = int(params.get('h') or 2)
        self.bars_before_fakeout = int(params.get('i') or 2)
        self.fakeout_perc_slack = float(params.get('j') or 0.5)
        self.slack_for_breakout_perc = float(params.get('k') or 0.1)
        
        self.upper_sr = 0
        self.lower_sr = 0
        self.prev_upper_sr_level = 0
        self.prev_lower_sr_level = 0
    
    def calculate_sr_levels(self, rates):
        """
        Calculate the Support and Resistance (SR) levels for the given rates.

        Parameters:
            rates  : Historical price data (OHLC).
        """
        highest_high = max([rate['high'] for rate in rates[-self.period_for_sr:]])
        lowest_low = min([rate['low'] for rate in rates[-self.period_for_sr:]])
        
        # Initialize upper and lower SR levels based on the current market conditions
        self.upper_sr = highest_high + (self.min_height_of_sr_distance * self.slack_for_sr_perc_div)
        self.lower_sr = lowest_low - (self.min_height_of_sr_distance * self.slack_for_sr_perc_div)
        
        # Store previous SR levels for Breakout and Fakeout checks
        self.prev_upper_sr_level = self.upper_sr
        self.prev_lower_sr_level = self.lower_sr

    def sr_check_is_buy(self, rates):
        """
        Check if the SR condition signals a buy based on proximity to lower SR.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if SR indicates a buy, False otherwise.
        """
        if self.lower_sr == 0:
            return False
        open_price = rates[-1]['open']
        return (open_price - self.lower_sr) < (self.max_distance_from_sr_perc * open_price)
    
    def sr_check_is_sell(self, rates):
        """
        Check if the SR condition signals a sell based on proximity to upper SR.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if SR indicates a sell, False otherwise.
        """
        if self.upper_sr == 0:
            return False
        open_price = rates[-1]['open']
        return (self.upper_sr - open_price) < (self.max_distance_from_sr_perc * open_price)

    def breakout_check_is_buy(self, rates):
        """
        Check if the Breakout condition signals a buy.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if Breakout indicates a buy, False otherwise.
        """
        if self.prev_upper_sr_level == 0:
            return False
        return rates[-2]['close'] > (self.prev_upper_sr_level + self.slack_for_breakout_perc * rates[-2]['close'])

    def breakout_check_is_sell(self, rates):
        """
        Check if the Breakout condition signals a sell.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if Breakout indicates a sell, False otherwise.
        """
        if self.prev_lower_sr_level == 0:
            return False
        return rates[-2]['close'] < (self.prev_lower_sr_level - self.slack_for_breakout_perc * rates[-2]['close'])

    def fakeout_check_is_buy(self, rates):
        """
        Check if the Fakeout condition signals a buy.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if Fakeout indicates a buy, False otherwise.
        """
        if self.lower_sr == 0:
            return False
        lowest_in_fakeout = min([rate['low'] for rate in rates[-self.bars_from_fakeout:]])
        previous_lowest = min([rate['low'] for rate in rates[-(self.bars_before_fakeout + self.bars_from_fakeout):-self.bars_from_fakeout]])
        return lowest_in_fakeout > (self.lower_sr - self.fakeout_perc_slack * lowest_in_fakeout) and previous_lowest < self.lower_sr

    def fakeout_check_is_sell(self, rates):
        """
        Check if the Fakeout condition signals a sell.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if Fakeout indicates a sell, False otherwise.
        """
        if self.upper_sr == 0:
            return False
        highest_in_fakeout = max([rate['high'] for rate in rates[-self.bars_from_fakeout:]])
        previous_highest = max([rate['high'] for rate in rates[-(self.bars_before_fakeout + self.bars_from_fakeout):-self.bars_from_fakeout]])
        return highest_in_fakeout < (self.upper_sr + self.fakeout_perc_slack * highest_in_fakeout) and previous_highest > self.upper_sr

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on SR, Breakout, and Fakeout conditions.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        # Check for SR buy/sell signals
        if self.sr_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_sr - 10, 'tp': self.upper_sr}
        if self.sr_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_sr + 10, 'tp': self.lower_sr}
        
        # Check for Breakout buy/sell signals
        if self.breakout_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_sr - 10, 'tp': self.upper_sr + 20}
        if self.breakout_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_sr + 10, 'tp': self.lower_sr - 20}
        
        # Check for Fakeout buy/sell signals
        if self.fakeout_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_sr - 10, 'tp': self.upper_sr}
        if self.fakeout_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_sr + 10, 'tp': self.lower_sr}
        
        return None, None




#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class TrendIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the Trend Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for trendline calculation.
        """
        super().__init__(params)
        self.period_for_trend = int(params.get('a', 100))
        self.slack_for_trend_perc = float(params.get('b', 0.1))
        self.min_slope_perc_div = float(params.get('c', 10))
        self.max_distance_from_trend_perc = float(params.get('d', 2))
        self.up_trend = {'line_start': 0, 'line_slope': 0, 'first_bar': 0}
        self.down_trend = {'line_start': 0, 'line_slope': 0, 'first_bar': 0}
        self.prev_up_trend = None
        self.prev_down_trend = None

    @staticmethod
    def calc_trend_value(slope, anchor_point, x):
        """
        Calculate the trendline value at a given point (x) based on the slope and anchor point.

        Parameters:
            slope (float): The slope of the trendline.
            anchor_point (float): The price at the trend's anchor point.
            x (int): The number of bars from the anchor.

        Returns:
            float: The calculated trendline value at the given x.
        """
        return slope * x + anchor_point

    def detect_up_trend(self, rates):
        """
        Detect an upward trend in the historical price data.

        Parameters:
            rates  : Historical price data (OHLC).
        """
        found = False
        for i in range(1, self.period_for_trend - 5):
            for k in range(self.period_for_trend, i + 5, -1):
                violated = False
                anchor_point = min(rates[k]['low'], rates[i]['low'])
                slope = (rates[i]['low'] - anchor_point) / (k - i)
                if slope < self.slack_for_trend_perc / self.min_slope_perc_div:
                    continue
                for l in range(k, 0, -1):
                    if rates[l]['close'] < self.calc_trend_value(slope, anchor_point, k - l):
                        violated = True
                        break
                if violated:
                    continue
                self.up_trend = {'line_start': anchor_point, 'line_slope': slope, 'first_bar': k}
                found = True
                break
            if found:
                break

    def detect_down_trend(self, rates):
        """
        Detect a downward trend in the historical price data.

        Parameters:
            rates  : Historical price data (OHLC).
        """
        found = False
        for i in range(1, self.period_for_trend - 5):
            for k in range(self.period_for_trend, i + 5, -1):
                violated = False
                anchor_point = max(rates[k]['high'], rates[i]['high'])
                slope = (anchor_point - rates[i]['high']) / (k - i)
                if slope > -(self.slack_for_trend_perc / self.min_slope_perc_div):
                    continue
                for l in range(k, 0, -1):
                    if rates[l]['close'] > self.calc_trend_value(slope, anchor_point, k - l):
                        violated = True
                        break
                if violated:
                    continue
                self.down_trend = {'line_start': anchor_point, 'line_slope': slope, 'first_bar': k}
                found = True
                break
            if found:
                break

    def check_is_buy(self, rates):
        """
        Check if a buy signal is generated based on the upward trendline.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if a buy signal is detected, False otherwise.
        """
        if self.up_trend['line_start'] == 0:
            return False
        trend_level = self.calc_trend_value(self.up_trend['line_slope'], self.up_trend['line_start'], self.up_trend['first_bar'])
        return rates[-1]['open'] > trend_level and (rates[-1]['open'] - trend_level) < self.max_distance_from_trend_perc * rates[-1]['open']

    def check_is_sell(self, rates):
        """
        Check if a sell signal is generated based on the downward trendline.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if a sell signal is detected, False otherwise.
        """
        if self.down_trend['line_start'] == 0:
            return False
        trend_level = self.calc_trend_value(self.down_trend['line_slope'], self.down_trend['line_start'], self.down_trend['first_bar'])
        return rates[-1]['open'] < trend_level and (trend_level - rates[-1]['open']) < self.max_distance_from_trend_perc * rates[-1]['open']

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on trendline conditions.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        self.detect_up_trend(rates)
        self.detect_down_trend(rates)

        if self.check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.up_trend['line_start'] - 10, 'tp': rates[-1]['close'] + 20}
        if self.check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.down_trend['line_start'] + 10, 'tp': rates[-1]['close'] - 20}

        return None, None
#


#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class TrendBreakoutIndicator(TrendIndicator):
    def __init__(self, params):
        """
        Initialize the Trend Breakout Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for trendline and breakout calculation.
        """
        super().__init__(params)
        self.slack_for_breakout = float(params.get('e', 0.1))

    def breakout_check_is_buy(self, rates):
        """
        Check if a breakout above the downtrend has occurred (buy signal).

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if a breakout above the downtrend is detected, False otherwise.
        """
        if self.prev_down_trend is None or self.prev_down_trend['line_start'] == 0:
            return False
        trend_level = self.calc_trend_value(self.prev_down_trend['line_slope'], self.prev_down_trend['line_start'], self.prev_down_trend['first_bar'])
        return rates[-2]['close'] > trend_level + (self.slack_for_breakout * rates[-2]['close'])

    def breakout_check_is_sell(self, rates):
        """
        Check if a breakout below the uptrend has occurred (sell signal).

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            bool: True if a breakout below the uptrend is detected, False otherwise.
        """
        if self.prev_up_trend is None or self.prev_up_trend['line_start'] == 0:
            return False
        trend_level = self.calc_trend_value(self.prev_up_trend['line_slope'], self.prev_up_trend['line_start'], self.prev_up_trend['first_bar'])
        return rates[-2]['close'] < trend_level - (self.slack_for_breakout * rates[-2]['close'])

    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on trendline breakouts.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        self.detect_up_trend(rates)
        self.detect_down_trend(rates)

        if self.breakout_check_is_buy(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': rates[-1]['low'] - 10, 'tp': rates[-1]['close'] + 20}
        if self.breakout_check_is_sell(rates):
            return 'sell', {'entry': rates[-1]['close'], 'sl': rates[-1]['high'] + 10, 'tp': rates[-1]['close'] - 20}

        return None, None


#TODO: update the logic to be "smoarter" and more efficient
#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class BarsTrendIndicator(TrendIndicator):
    """
    Make a trade decision based on breaking the highest/lowest bars in a trend.
    trend is determined by the x bars lookback period.
    Parameters:
        rates  : Historical price data (OHLC).

    Returns:
        tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
    """

    def __init__(self, params):
        """
        Initialize the Bars Trend Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for trendline and breakout calculation.
        """
        super().__init__(params)
        self.bars_lookback_period = int(params.get('a', 10))


    def claculuate_and_make_make_trade_decision(self, rates):
        """
        Make a trade decision based on trendline breakouts.

        Parameters:
            rates  : Historical price data (OHLC).

        Returns:
            Tuple: ('buy'/'sell'/None, trade_data) - Where trade_data contains entry, SL, TP.
        """
        lowest_close = min([r['close'] for r in rates[-self.bars_lookback_period:]])
        highest_close = max([r['close'] for r in rates[-self.bars_lookback_period:]])
        if rates[-1]['close'] < lowest_close:
            return 'buy', {'entry': rates[-1]['close'], 'sl': lowest_close, 'tp': highest_close}
        elif rates[-1]['close'] > highest_close:
            return 'sell', {'entry': rates[-1]['close'], 'sl': highest_close, 'tp': lowest_close}

        return None, None









#TODO: Implement return trade values for the following classe
#TODO: run the strategy and check logic agains backtesting data and charts
###################################
###                             ###
### This needs a lot more work  ###
###                             ###
###################################
class SpecialIndicator(Indicator):
    def __init__(self, params):
        """
        Initialize the SpecialIndicator with parameters.
        
        Parameters:
            params (dict): A dictionary containing parameters for the Special Indicator.
        """
        self.special_id = params.get('a', 1)  # Indicator parameter 'a' defines the special strategy.
        self.xbars_special = params.get('b', 5)
        self.xbars_2nd_special = params.get('c', 5)
        self.special_period = params.get('d', 3)
        self.special_multiplier_entry = params.get('e', 0.01)
        self.special_multiplier_sl = params.get('f', 2.0)

    def calculate_trade_decision(self, rates):
        """
        Determine trade decision based on the Special indicator strategy.

        Parameters:
            rates  : The OHLC data.

        Returns:
            Tuple: (decision, trade_data) - where decision is a string ('buy', 'sell', None)
                   and trade_data is a dictionary containing trade parameters (entry, stop loss, take profit).
        """
        # Dynamically call the appropriate method based on special_id
        method_name = f"special_{self.special_id}_calculate_trade_decision"
        method = getattr(self, method_name, None)
        if callable(method):
            return method(rates)
        return None, None


    #TODO: 100 is identical to bar trend indicator, replace with other?
    # --- Special Strategy 100 ---
    def special_100_calculate_trade_decision(self, rates):
        lowest_close = min([r['close'] for r in rates[-self.xbars_special-1:-1]])
        highest_close = max([r['close'] for r in rates[-self.xbars_special-1:-1]])
        if rates[-1]['close'] < lowest_close:
            return 'buy', {'entry': rates[-1]['close'], 'sl': lowest_close, 'tp': highest_close}
        elif rates[-1]['close'] > highest_close:
            return 'sell', {'entry': rates[-1]['close'], 'sl': highest_close, 'tp': lowest_close}
        return None, None

    # --- Special Strategy 101 ---
    def special_101_calculate_trade_decision(self, rates):
        if self.in_bar(rates):
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.calculate_candle_size(rates[-1]), 'tp': None}
        return None, None

    def in_bar(self, rates):
        current = rates.iloc[-1]
        previous = rates.iloc[-2]
        comparison = rates.iloc[-3]
        return current['high'] < previous['high'] and current['low'] > comparison['low']

    @staticmethod
    def calculate_candle_size(bar):
        # Placeholder for calculating candle size
        return abs(bar['high'] - bar['low'])

    # --- Special Strategy 102 ---
    def special_102_calculate_trade_decision(self, rates):
        long_ma = self.calculate_long_ma(rates)
        if rates[-1]['close'] < (1 - self.special_multiplier_entry) * long_ma:
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.special_multiplier_sl * long_ma, 'tp': None}
        elif rates[-1]['close'] > (1 + self.special_multiplier_entry) * long_ma:
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.special_multiplier_sl * long_ma, 'tp': None}
        return None, None

    def calculate_long_ma(self, rates):
        return sum([r['close'] for r in rates[-self.special_period:]]) / self.special_period

    # --- Special Strategy 103 ---
    def special_103_calculate_trade_decision(self, rates):
        for i in range(2, self.xbars_special + 2):
            if not self.hhhc(rates, i):
                return None, None
        if self.lllc(rates, 1):
            return 'buy', {'entry': rates[-1]['close'], 'sl': None, 'tp': None}
        elif self.hhhc(rates, 1):
            return 'sell', {'entry': rates[-1]['close'], 'sl': None, 'tp': None}
        return None, None

    @staticmethod
    def hhhc(rates, i):
        return rates[i]['close'] > rates[i+1]['close'] and rates[i]['high'] > rates[i+1]['high']
    @staticmethod
    def lllc(rates, i):
        return rates[i]['close'] < rates[i+1]['close'] and rates[i]['low'] < rates[i+1]['low']


    #TODO: 104 is based on BB - maybe add it to the BB indicator?
    # --- Special Strategy 104 ---
    def special_104_calculate_trade_decision(self, rates):
        if rates[-1]['close'] > self.upper_band(rates, -1) and rates[-2]['close'] > self.upper_band(rates, -2) and rates[-3]['close'] < self.upper_band(rates, -3):
            return 'buy', {'entry': rates[-1]['close'], 'sl': self.lower_band(rates, -1), 'tp': self.upper_band(rates, -1)}
        elif rates[-1]['close'] < self.lower_band(rates, -1) and rates[-2]['close'] < self.lower_band(rates, -2) and rates[-3]['close'] > self.lower_band(rates, -3):
            return 'sell', {'entry': rates[-1]['close'], 'sl': self.upper_band(rates, -1), 'tp': self.lower_band(rates, -1)}
        return None, None

    def upper_band(self, rates, index):
        # Placeholder for upper band calculation
        return rates[index]['close'] * 1.05

    def lower_band(self, rates, index):
        # Placeholder for lower band calculation
        return rates[index]['close'] * 0.95

    # --- Special Strategy 105 ---
    def special_105_calculate_trade_decision(self, rates):
        #placeholder logic for special strategy 105
        return None, None

    # --- Special Strategy 106 ---
    def special_106_calculate_trade_decision(self, rates):
        #placeholder logic for special strategy 106
        return None, None
    
    # --- Special Strategy 107 ---
    def special_107_calculate_trade_decision(self, rates):
        #placeholder logic for special strategy 107
        return None, None