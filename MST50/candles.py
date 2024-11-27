"""
This module contains the `CandlePatterns` class that handles candle patterns and their trade decisions.

The `CandlePatterns` class is initialized with configurations for a specific timeframe.
It contains methods to make trade decisions based on the configured candle patterns.

Classes:
    - CandlePatterns: Handles the candle patterns and their trade decisions.
    - Pattern: Base class for candle patterns.
        - Child Classes:
            - NoPattern
            - Engulf
            - Marubozu
            - Out
            - InPattern
            - Hammer
            - InvertedHammer
            - KangarooTailFull
            - KangarooTailPartial
            - Fakeout
            - SameCandleCount
            - InsideBreakout
            - HHHCLLLC
            - InvHHHCLLLC
    - Candle: Base class for individual candle analysis.
        - Child Classes:
            - NoCandle
            - SameDirectionCandle
            - OppositeDirectionCandle
            - DojiCandle
"""

import numpy as np
from .constants import CandleColor


class CandlePatterns:
    """
    Class to handle the candle patterns and their trade decisions.
    """

    def __init__(self, candles_config: dict):
        """
        Initialize the CandlePatterns class with configurations for a specific timeframe.

        Parameters:
            candles_config (dict): Dictionary containing configurations for the timeframe.
        """
        self.candles_config = candles_config
        self.pattern_name = candles_config.get('barsP_pattern', '0')
        self.pattern_candles_count = candles_config.get('barsP_pattern_count', 1)

        # Map pattern names to their corresponding classes
        self.pattern_mapping = {
            '0': NoPattern,  # No pattern
            'Engulf': Engulf,
            'Marubozu': Marubozu,
            'Out': Out,
            'In': InPattern,
            'Ham': Hammer,
            'InvHam': InvertedHammer,
            'Kangoro_full': KangarooTailFull,
            'Kangoro_partial': KangarooTailPartial,
            'Fakeout': Fakeout,
            'Same_Candle_Count': SameCandleCount,
            'Inside_breakout': InsideBreakout,
            'HHHCLLLC': HHHCLLLC,
            'Inv_HHHCLLLC': InvHHHCLLLC,
        }
        self.candle_mapping = {
            '0': NoCandle,  # No candle
            'same_direction': SameDirectionCandle,
            'opposite_direction': OppositeDirectionCandle,
            'doji': DojiCandle,
        }

        # Initialize the appropriate candle pattern class based on candles config
        self.candle_pattern_name = candles_config.get('barsP_pattern', '0')
        self.candle_pattern_instance = self.initialize_pattern()

        # Candles
        self.first_candle_instance = self.initialize_candle(candles_config.get('barsP_1st_candle', '0'))
        self.second_candle_instance = self.initialize_candle(candles_config.get('barsP_2nd_candle', '0'))
        self.third_candle_instance = self.initialize_candle(candles_config.get('barsP_3rd_candle', '0'))

    def initialize_pattern(self):
        """
        Initialize the appropriate candle pattern class based on the configured pattern.

        Returns:
            Pattern: Candle pattern class instance.
        """
        if self.candle_pattern_name in self.pattern_mapping:
            candle_pattern_class = self.pattern_mapping[self.candle_pattern_name]
            return candle_pattern_class(self.pattern_candles_count)
        return None  # Invalid pattern name

    def initialize_candle(self, position_candle: str):
        """
        Initialize the appropriate candle class based on the configured candle.

        Returns:
            Candle: Candle class instance.
        """
        if position_candle in self.candle_mapping:
            candle_class = self.candle_mapping[position_candle]
            return candle_class()
        return None  # Invalid candle name

    def make_trade_decision(self, rates: np.recarray) -> str:
        """
        Make a trade decision based on the configured candle patterns.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str or None: 'buy', 'sell', 'both' or None based on the aggregated decisions.
        """
        decisions = []
        if self.candle_pattern_instance:
            decision = self.candle_pattern_instance.calculate_and_make_trade_decision(rates)
            if not decision:
                return None  # No valid decisions from the pattern, exit early
            decisions.append(decision)

        # Handle individual candle conditions if defined
        candles = [
            candle for candle in [
                self.first_candle_instance,
                self.second_candle_instance,
                self.third_candle_instance
            ] if candle is not None and candle.__class__ != NoCandle
        ]
        for candle in candles:
            decision = candle.calculate_and_make_trade_decision(rates)
            if not decision:
                return None  # No valid decisions from the candle, exit early
            decisions.append(decision)

        unique_decisions = set(decisions)
        if len(unique_decisions) == 1:
            return unique_decisions.pop()
        return None  # Conflicting decisions

    def check_candle_patterns_active(self) -> bool:
        """
        Check if the configured candle pattern is active.

        Returns:
            bool: True if the pattern is active, False otherwise.
        """
        pattern_active = self.pattern_name != '0' and self.candle_pattern_instance is not None
        first_candle_active = self.first_candle_instance is not None and self.first_candle_instance.__class__ != NoCandle
        second_candle_active = self.second_candle_instance is not None and self.second_candle_instance.__class__ != NoCandle
        third_candle_active = self.third_candle_instance is not None and self.third_candle_instance.__class__ != NoCandle

        return any([pattern_active, first_candle_active, second_candle_active, third_candle_active])


class Pattern:
    """
    Base class for candle patterns.

    Parameters:
        pattern_candles_count (int): Number of candles required for the pattern.
    """

    def __init__(self, pattern_candles_count: int):
        self.pattern_candles_count = pattern_candles_count
        self.trade_decision_method = None  # Will be set by the child classes

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Placeholder method for calculating the pattern based on the provided rates.
        Should be overridden by child classes.

        Returns:
            bool: True if pattern is detected, False otherwise.
        """
        raise NotImplementedError("Pattern calculation method not implemented")

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        """
        Calculate the pattern and make a trade decision based on the pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str or None: 'buy', 'sell', 'both' or None based on the pattern.
        """
        if self.trade_decision_method:
            return self.trade_decision_method(rates)
        return None  # No trade decision method set

    def upper_wick_size(self, rates: np.recarray, candle_i: int) -> float:
        """
        Calculate the upper wick size of a candle.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle (1-based, where 1 is the latest candle).

        Returns:
            float: Upper wick size.
        """
        candle = rates[-candle_i]
        if Candle.candle_color(candle) == -1:  # Red candle
            return candle['high'] - candle['open']
        else:
            return candle['high'] - candle['close']

    def lower_wick_size(self, rates: np.recarray, candle_i: int) -> float:
        """
        Calculate the lower wick size of a candle.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle (1-based, where 1 is the latest candle).

        Returns:
            float: Lower wick size.
        """
        candle = rates[-candle_i]
        if Candle.candle_color(candle) == -1:  # Red candle
            return candle['close'] - candle['low']
        else:
            return candle['open'] - candle['low']

    def upper_wick_ratio(self, rates: np.recarray, candle_i: int) -> float:
        """
        Calculate the upper wick ratio of a candle.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle.

        Returns:
            float: Upper wick ratio.
        """
        size = self.upper_wick_size(rates, candle_i)
        body = self.body_size(rates, candle_i)
        return body / size if size != 0 else 1000.0

    def lower_wick_ratio(self, rates: np.recarray, candle_i: int) -> float:
        """
        Calculate the lower wick ratio of a candle.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle.

        Returns:
            float: Lower wick ratio.
        """
        size = self.lower_wick_size(rates, candle_i)
        body = self.body_size(rates, candle_i)
        return body / size if size != 0 else 1000.0

    def wick_ratio(self, rates: np.recarray, candle_i: int) -> float:
        """
        Calculate the overall wick ratio of a candle.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle.

        Returns:
            float: Wick ratio.
        """
        upper = self.upper_wick_size(rates, candle_i)
        lower = self.lower_wick_size(rates, candle_i)
        total_wick = upper + lower
        if total_wick == 0:
            return 10.0  # No wicks
        return self.body_size(rates, candle_i) / total_wick

    @staticmethod
    def hhhc(rates: np.recarray, i: int) -> bool:
        """
        Higher High Higher Close pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            i (int): Index of the candle.

        Returns:
            bool: True if pattern is detected, False otherwise.
        """
        return rates[-i]['close'] > rates[-i - 1]['close'] and rates[-i]['high'] > rates[-i - 1]['high']

    @staticmethod
    def lllc(rates: np.recarray, i: int) -> bool:
        """
        Lower Low Lower Close pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            i (int): Index of the candle.

        Returns:
            bool: True if pattern is detected, False otherwise.
        """
        return rates[-i]['close'] < rates[-i - 1]['close'] and rates[-i]['low'] < rates[-i - 1]['low']

    @staticmethod
    def hhhl(rates: np.recarray, i: int) -> bool:
        """
        Higher High Higher Low pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            i (int): Index of the candle.

        Returns:
            bool: True if pattern is detected, False otherwise.
        """
        return rates[-i]['high'] > rates[-i - 1]['high'] and rates[-i]['low'] > rates[-i - 1]['low']

    @staticmethod
    def lhll(rates: np.recarray, i: int) -> bool:
        """
        Lower High Lower Low pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            i (int): Index of the candle.

        Returns:
            bool: True if pattern is detected, False otherwise.
        """
        return rates[-i]['high'] < rates[-i - 1]['high'] and rates[-i]['low'] < rates[-i - 1]['low']

    @staticmethod
    def same_candle_count(rates: np.recarray, candle_i: int) -> int:
        """
        Count the number of consecutive candles with the same color.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle (1-based, where 1 is the latest candle).

        Returns:
            int: Number of consecutive candles with the same color.
        """
        base_color = Candle.candle_color(rates[-candle_i])
        if base_color == 0:
            return 1
        count = 1
        while (candle_i + count) <= len(rates):
            current_color = Candle.candle_color(rates[-(candle_i + count)])
            if current_color == base_color:
                count += 1
            else:
                break
        return count

    @staticmethod
    def body_size(rates: np.recarray, candle_i: int) -> float:
        """
        Calculate the body size of a candle.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).
            candle_i (int): Index of the candle.

        Returns:
            float: Absolute body size.
        """
        candle = rates[-candle_i]
        return abs(candle['close'] - candle['open'])


class NoPattern(Pattern):
    """
    A placeholder class for no pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.no_trade_decision

    def calculate_pattern(self, rates: np.recarray) -> bool:
        return False

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        return None

    def no_trade_decision(self, rates: np.recarray) -> str:
        return None


class Engulf(Pattern):
    """
    Engulfing pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.engulf

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Engulfing pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Engulfing pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates[-1])
            previous_color = Candle.candle_color(rates[-2])
            if current_color == previous_color or current_color == 0 or previous_color == 0:
                return False

            current_open = rates[-1]['open']
            current_close = rates[-1]['close']
            previous_open = rates[-2]['open']
            previous_close = rates[-2]['close']

            if current_color == 1 and current_open <= previous_close and current_close >= previous_open:
                return True
            elif current_color == -1 and current_open >= previous_close and current_close <= previous_open:
                return True
            return False
        except IndexError:
            return False

    def engulf(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Engulfing pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        if self.calculate_pattern(rates, None):
            current_color = Candle.candle_color(rates[-1])
            if current_color == 1:
                return 'buy'
            elif current_color == -1:
                return 'sell'
        return None


class Marubozu(Pattern):
    """
    Marubozu pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.marubozu

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Marubozu pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Marubozu pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates[-1])
            wick_ratio = self.wick_ratio(rates, 1)
            upper_wick_ratio = self.upper_wick_ratio(rates, 1)
            lower_wick_ratio = self.lower_wick_ratio(rates, 1)

            if current_color == 0:
                return False
            if wick_ratio > 1.75 and max(upper_wick_ratio, lower_wick_ratio) >= 3:
                return True
            return False
        except IndexError:
            return False

    def marubozu(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Marubozu pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        if self.calculate_pattern(rates, None):
            current_color = Candle.candle_color(rates[-1])
            if current_color == 1:
                return 'buy'
            elif current_color == -1:
                return 'sell'
        return None


class Out(Pattern):
    """
    Outside Bar pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.out_pattern

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Outside Bar pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Outside Bar pattern is detected, False otherwise.
        """
        try:
            current = rates[-1]
            previous = rates[-2]
            if current['high'] > previous['high'] and current['low'] < previous['low']:
                return True
            return False
        except IndexError:
            return False

    def out_pattern(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Outside Bar pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            current = rates[-1]
            previous = rates[-2]
            return 'both' if current['close'] > previous['close'] else 'sell'
        return None


class InPattern(Pattern):
    """
    Inside Bar pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.in_pattern

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Inside Bar pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Inside Bar pattern is detected, False otherwise.
        """
        try:
            current = rates[-1]
            previous = rates[-2]
            if current['high'] < previous['high'] and current['low'] > previous['low']:
                return True
            return False
        except IndexError:
            return False

    def in_pattern(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Inside Bar pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class Hammer(Pattern):
    """
    Hammer pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.hammer

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Hammer pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Hammer pattern is detected, False otherwise.
        """
        try:
            if self.wick_ratio(rates, 1) > 0.3:
                return False
            if self.upper_wick_size(rates, 1) == 0:
                return True
            if self.lower_wick_size(rates, 1) / self.upper_wick_size(rates, 1) > 2:
                return True
            return False
        except IndexError:
            return False

    def hammer(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Hammer pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class InvertedHammer(Pattern):
    """
    Inverted Hammer pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.inverted_hammer

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Inverted Hammer pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Inverted Hammer pattern is detected, False otherwise.
        """
        try:
            if self.wick_ratio(rates, 1) > 0.3:
                return False
            if self.lower_wick_size(rates, 1) == 0:
                return True
            if self.upper_wick_size(rates, 1) / self.lower_wick_size(rates, 1) > 2:
                return True
            return False
        except IndexError:
            return False

    def inverted_hammer(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Inverted Hammer pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class KangarooTailFull(Pattern):
    """
    Kangaroo Tail Full pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.kangaroo_tail_full

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Kangaroo Tail Full pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Kangaroo Tail Full pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates[-1])
            if current_color == 1:
                return self.lhll(rates, 1) and self.hhhl(rates, 1)
            elif current_color == -1:
                return self.hhhl(rates, 1) and self.lhll(rates, 1)
            return False
        except IndexError:
            return False

    def kangaroo_tail_full(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Kangaroo Tail Full pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class KangarooTailPartial(Pattern):
    """
    Kangaroo Tail Partial pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.kangaroo_tail_partial

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Kangaroo Tail Partial pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Kangaroo Tail Partial pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates[-1])
            if current_color == 1:
                return self.lhll(rates, 1)
            elif current_color == -1:
                return self.hhhl(rates, 1)
            return False
        except IndexError:
            return False

    def kangaroo_tail_partial(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Kangaroo Tail Partial pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class Fakeout(Pattern):
    """
    Fakeout pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.fakeout

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Fakeout pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Fakeout pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates[-1])
            if current_color == 1 and rates[-1]['close'] < rates[-2]['high']:
                return True
            elif current_color == -1 and rates[-1]['close'] > rates[-2]['low']:
                return True
            return False
        except IndexError:
            return False

    def fakeout(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Fakeout pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class SameCandleCount(Pattern):
    """
    Same Candle Count pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.same_candle_count_pattern

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Same Candle Count pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Same Candle Count pattern is detected, False otherwise.
        """
        if self.same_candle_count(rates, 1) >= self.pattern_candles_count:
            return True
        return False

    def same_candle_count_pattern(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Same Candle Count pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class InsideBreakout(Pattern):
    """
    Inside Breakout pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.inside_breakout

    def calculate_pattern(self, rates: np.recarray) -> bool:
        """
        Check for Inside Breakout pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            bool: True if Inside Breakout pattern is detected, False otherwise.
        """
        try:
            in_bar_pattern = InPattern(self.pattern_candles_count)
            if not in_bar_pattern.calculate_pattern(rates):
                return False
            if rates[-2]['close'] > rates[-4]['high']:
                return True
            elif rates[-2]['close'] < rates[-4]['low']:
                return True
            return False
        except IndexError:
            return False

    def inside_breakout(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Inside Breakout pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates, None):
            return 'both'
        return None


class HHHCLLLC(Pattern):
    """
    Higher High Higher Close / Lower Low Lower Close pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.hhhclllc

    def calculate_pattern(self, rates: np.recarray) -> bool:
        # Not required for this pattern
        pass

    def hhhclllc(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on HHHCLLLC pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        if self.hhhc(rates, 1):
            return 'buy'
        if self.lllc(rates, 1):
            return 'sell'
        return None


class InvHHHCLLLC(Pattern):
    """
    Inverted HHHCLLLC pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.inv_hhhclllc

    def calculate_pattern(self, rates: np.recarray) -> bool:
        # Not required for this pattern
        pass

    def inv_hhhclllc(self, rates: np.recarray) -> str:
        """
        Determine trade decision based on Inverted HHHCLLLC pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        if self.hhhc(rates, 1):
            return 'sell'
        if self.lllc(rates, 1):
            return 'buy'
        return None


class Candle:
    """
    Base class for candle analysis.
    """

    def __init__(self):
        pass

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        """
        A placeholder method for calculating the candle pattern and making a trade decision.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        raise NotImplementedError("Candle calculation method not implemented")

    @staticmethod
    def candle_color(candle) -> int:
        """
        Determine the color of a specific candle.

        Parameters:
            candle (np.record): A single candle's data.

        Returns:
            int: 1 for green (bullish), -1 for red (bearish), 0 for doji.
        """
        if candle['close'] > candle['open']:
            return 1  # Green (bullish)
        elif candle['close'] < candle['open']:
            return -1  # Red (bearish)
        else:
            return 0  # Doji (indecision)


class NoCandle(Candle):
    """
    Placeholder class for no candle.
    """

    def __init__(self):
        super().__init__()

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        return None


class SameDirectionCandle(Candle):
    """
    Same Direction Candle class.
    """

    def __init__(self):
        super().__init__()

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        """
        Check for Same Direction Candle pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        color = Candle.candle_color(rates[-1])
        if color == 1:
            return 'buy'
        elif color == -1:
            return 'sell'
        return None


class OppositeDirectionCandle(Candle):
    """
    Opposite Direction Candle class.
    """

    def __init__(self):
        super().__init__()

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        """
        Check for Opposite Direction Candle pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        color = Candle.candle_color(rates[-1])
        if color == 1:
            return 'sell'
        elif color == -1:
            return 'buy'
        return None


class DojiCandle(Candle):
    """
    Doji Candle class.
    """

    def __init__(self):
        super().__init__()

    def calculate_and_make_trade_decision(self, rates: np.recarray) -> str:
        """
        Check for Doji Candle pattern.

        Parameters:
            rates (np.recarray): Historical price data (OHLC).

        Returns:
            str: 'both' if doji is detected, None otherwise.
        """
        color = Candle.candle_color(rates[-1])
        if color == 0:
            return 'both'
        return None