# signals.py

import numpy as np
import pandas as pd
from datetime import datetime

# Import utility functions and constants
from .constants import DEVIATION, TRADE_DIRECTION



class RSISignal:
    def __init__(self, rsi_period=14,max_deviation=30, min_deviation=30):
        self.rsi_period = int(rsi_period)
        self.max_deviation = int(max_deviation)
        self.min_deviation = int(min_deviation)

    def calculate_rsi(self, rates):
        """
        Calculate the RSI based on the given OHLC data (using the standard RSI calculation).

        Parameters:
            rates (df): Historical price data (OHLC).

        Returns:
            list: Calculated RSI values.
        """
        # Extract close prices from rates
        close_prices = rates['close'].values
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
        """
        Calculate the RSI based on the given OHLC data (using the standard RSI calculation).

        Parameters:
            rates (list): Historical price data (OHLC).
        
        Returns:
            list: Calculated RSI values.
        """
        if len(close_prices) < self.rsi_period + 1:
        # Not enough data to calculate RSI
            return []
        close_prices = [rate['close'] for rate in rates]

        # Calculate price changes
        changes = [close_prices[i] - close_prices[i - 1] for i in range(1, len(close_prices))]

        # Calculate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [-min(change, 0) for change in changes]

        # Initialize the average gain and loss
        avg_gain = sum(gains[:self.rsi_period]) / self.rsi_period
        avg_loss = sum(losses[:self.rsi_period]) / self.rsi_period

        # Calculate RSI values
        rsi_values = []
        for i in range(self.rsi_period, len(close_prices)):
            avg_gain = ((avg_gain * (self.rsi_period - 1)) + gains[i - 1]) / self.rsi_period
            avg_loss = ((avg_loss * (self.rsi_period - 1)) + losses[i - 1]) / self.rsi_period

            # Calculate RSI
            if avg_loss == 0:
                rsi = 100  # If there are no losses, RSI is 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

        return rsi_values

    def check_rsi_signal(self, last_rsi_value, trade_direction):
        """
        Check the RSI signal based on maximum and minimum deviation levels.

        Parameters:
            rsi_values (list): List of calculated RSI values.
            max_deviation (float): Maximum allowed RSI deviation.
            min_deviation (float): Minimum allowed RSI deviation.

        Returns:
            bool: True if RSI is within acceptable deviation limits, False otherwise.
        """

        effective_rsi_value = last_rsi_value if trade_direction == 'buy' else 100 - last_rsi_value
        if effective_rsi_value > 50+ self.max_deviation or effective_rsi_value < 50 - self.min_deviation: # overbought or oversold
            return False
        return True  # Within acceptable range





class ERSignal():
    def __init__(self, low_exit_value=0, high_exit_value=1):
        """
        ER Signal class for use as an indicator filter.
        Initialize the ER Indicator with strategy parameters.

        Parameters:
            params (dict): Contains specific parameters for ER Ratio calculation.
        """
        super().__init__()
        self.low_exit_value = low_exit_value
        self.high_exit_value = high_exit_value

    def calculate_er_ratio(self, rates):
        """
        Calculate the Efficiency Ratio (ER) based on the given candles.

        Parameters:
            rates (list): Historical price data (OHLC).

        Returns:
            float: The Efficiency Ratio.
        """
        change = abs(rates[-1]['close'] - rates[-(self.candles_count + 1)]['close'])
        derivative = sum(abs(rates[-i]['close'] - rates[-(i+1)]['close']) for i in range(1, self.candles_count + 1))
        if derivative == 0:
            return 0.0  # Avoid division by zero
        return change / derivative

    #TODO: Implement the following methods
    def manage_high_noise_exit(self, position_type, er_ratio, last_candle_color):
        """
        Manage exits in high noise conditions based on ER.

        Parameters:
            position (dict): The current open position.
            rates (list): Historical price data (OHLC).
        
        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        return False

    #TODO: Implement the following methods
    def manage_low_noise_against_exit(self, position_type, er_ratio, last_candle_color):
        """
        Manage exits when there's low noise but the trade is against the trend.

        Parameters:
            position (dict): The current open position.
            rates (list): Historical price data (OHLC).
        
        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        return False

    #TODO: Implement the following methods
    def er_signal_check(self, position, rates):
        """
        Check the ER signal for position management.

        Parameters:
            position (dict): The current open position.
            rates (list): Historical price data (OHLC).
        
        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        er_ratio = self.calculate_er_ratio(rates)
        last_candle_color = self.candle_color(rates[-1])
        position_type = position['type']
        return False

