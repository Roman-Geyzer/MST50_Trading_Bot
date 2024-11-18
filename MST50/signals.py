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

# Import the ta library for technical analysis
import ta


class RSISignal:
    def __init__(self, rsi_period=14, max_deviation=30, min_deviation=30):
        """
        Initialize the RSISignal class with strategy parameters.

        Parameters:
            rsi_period (int): The period over which to calculate RSI.
            max_deviation (int): The maximum deviation from the midpoint for buy signals.
            min_deviation (int): The minimum deviation from the midpoint for sell signals.
        """
        self.rsi_period = int(rsi_period)
        self.max_deviation = int(max_deviation)
        self.min_deviation = int(min_deviation)


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
    



class ERSignal:
    def __init__(self, candles_count=14, low_exit_value=0, high_exit_value=1):
        """
        ER Signal class for use as an indicator filter.
        Initialize the ER Indicator with strategy parameters.

        Parameters:
            candles_count (int): The number of candles to consider for ER calculation.
            low_exit_value (int): Threshold for low noise exit.
            high_exit_value (int): Threshold for high noise exit.
        """
        self.candles_count = int(candles_count)
        self.low_exit_value = low_exit_value
        self.high_exit_value = high_exit_value

    def calculate_er_ratio(self, rates):
        """
        Calculate the Efficiency Ratio (ER) based on the given candles.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            float: The Efficiency Ratio.
        """
        if len(rates) < self.candles_count + 1:
            return 0.0  # Avoid division by zero or insufficient data

        change = abs(rates['close'][-1] - rates['close'][-(self.candles_count + 1)])
        derivative = sum(abs(rates['close'][-i] - rates['close'][-(i + 1)]) for i in range(1, self.candles_count + 1))
        if derivative == 0:
            return 0.0  # Avoid division by zero
        return change / derivative

    def candle_color(self, candle):
        """
        Determine the color of the candle.

        Parameters:
            candle (dict): A single candle's OHLC data.

        Returns:
            str: 'green' if bullish, 'red' if bearish, 'neutral' otherwise.
        """
        if candle['close'] > candle['open']:
            return 'green'
        elif candle['close'] < candle['open']:
            return 'red'
        else:
            return 'neutral'

    def manage_high_noise_exit(self, position_type, er_ratio, last_candle_color):
        """
        Manage exits in high noise conditions based on ER.

        Parameters:
            position_type (str): The type of the current open position ('buy' or 'sell').
            er_ratio (float): The current Efficiency Ratio.
            last_candle_color (str): The color of the last candle ('green', 'red', 'neutral').

        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        if position_type == 'buy' and er_ratio > self.high_exit_value and last_candle_color == 'red':
            return True
        elif position_type == 'sell' and er_ratio > self.high_exit_value and last_candle_color == 'green':
            return True
        return False

    def manage_low_noise_against_exit(self, position_type, er_ratio, last_candle_color):
        """
        Manage exits when there's low noise but the trade is against the trend.

        Parameters:
            position_type (str): The type of the current open position ('buy' or 'sell').
            er_ratio (float): The current Efficiency Ratio.
            last_candle_color (str): The color of the last candle ('green', 'red', 'neutral').

        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        if position_type == 'buy' and er_ratio < self.low_exit_value and last_candle_color == 'red':
            return True
        elif position_type == 'sell' and er_ratio < self.low_exit_value and last_candle_color == 'green':
            return True
        return False

    def er_signal_check(self, position, rates):
        """
        Check the ER signal for position management.

        Parameters:
            position (dict): The current open position with keys like 'type'.
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if the position should be closed, False otherwise.
        """
        er_ratio = self.calculate_er_ratio(rates)
        last_candle = rates[-1]
        last_candle_color = self.candle_color(last_candle)
        position_type = position.get('type', '').lower()

        if self.manage_high_noise_exit(position_type, er_ratio, last_candle_color):
            return True

        if self.manage_low_noise_against_exit(position_type, er_ratio, last_candle_color):
            return True

        return False