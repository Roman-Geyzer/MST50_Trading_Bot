# signals.py
"""
This module provides signal classes for use as indicator filters.
Classes:
    RSISignal: Class to check RSI signals for trade entry.
    ERSignal: Class to check ER signals for trade entry.
"""

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
        Calculate the RSI using numpy for better performance.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            np.ndarray: Calculated RSI values.
        """
        close_prices = rates['close']
        if len(close_prices) < self.rsi_period + 1:
            return np.array([])  # Return an empty array if not enough data

        deltas = np.diff(close_prices)
        seed = deltas[:self.rsi_period]
        up = seed[seed >= 0].sum() / self.rsi_period
        down = -seed[seed < 0].sum() / self.rsi_period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(close_prices)
        rsi[:self.rsi_period] = 100. - 100. / (1. + rs)

        for i in range(self.rsi_period, len(close_prices)):
            delta = deltas[i - 1]  # The diff is 1 shorter

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (self.rsi_period - 1) + upval) / self.rsi_period
            down = (down * (self.rsi_period - 1) + downval) / self.rsi_period

            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    def check_rsi_signal(self, last_rsi_value, trade_direction):
        """
        Check the RSI signal based on maximum and minimum deviation levels.

        Parameters:
            last_rsi_value (float): The latest RSI value.
            trade_direction (str): The trade direction ('buy' or 'sell').

        Returns:
            bool: True if RSI is within acceptable deviation limits, False otherwise.
        """
        if trade_direction == 'buy':
            effective_rsi_value = last_rsi_value
        else:
            effective_rsi_value = 100 - last_rsi_value

        if effective_rsi_value > 50 + self.max_deviation or effective_rsi_value < 50 - self.min_deviation:
            # Overbought or oversold
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

