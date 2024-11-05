# candles.py
"""
This module contains the CandlePatterns class that handles the candle patterns and their trade decisions.
The CandlePatterns class is initialized with configurations for a specific timeframe.
The class contains methods to make trade decisions based on the configured candle patterns.
functions:
    CandlePatterns: Initialize the CandlePatterns class with configurations for a specific timeframe.
    initialize_pattern: Initialize the appropriate candle pattern class based on the configured pattern.
    initialize_candle: Initialize the appropriate candle class based on the configured candle.
    make_trade_decision: Make a trade decision based on the configured candle patterns.
    check_candle_patterns_active: Check if the configured candle pattern is active.
classes:
    CandlePatterns: Class to handle the candle patterns and their trade decisions.
    Pattern: Base class for candle patterns.
    Pattern Class children:
        NoPattern: Placeholder class for no pattern.
        Engulf: Initialize the Engulfing pattern.
        Marubuzo: Initialize the Marubuzo pattern.
        Out: Initialize the Out pattern.
        In: Initialize the InBar pattern.
        Ham: Initialize the Hammer pattern.
        InvHam: Initialize the Inverted Hammer pattern.
        KangoroFull: Initialize the Full Kangoro pattern.
        KangoroPartial: Initialize the Partial Kangoro pattern.
        Fakeout: Initialize the Fakeout pattern.
        SameCandleCount: Initialize the Same Candle Count pattern.
        InsideBreakout: Initialize the Inside Breakout pattern.
        HHHCLLLC: Initialize the HHHCLLLC pattern.
        InvHHHCLLLC: Initialize the Inverted HHHCLLLC pattern.
    Candle: Base class for candle patterns.
    Candle Class children:
        NoCandle: Placeholder class for no candle.
        SameDirectionCandle: Initialize the Same Direction Candle class.
        OppositeDirectionCandle: Initialize the Opposite Direction Candle class.
"""
import pandas as pd
from .constants import CandleColor
from .utils import print_with_info
from .plotting import plot_bars



class CandlePatterns:
    def __init__(self, candles_config: dict):
        """
        Initialize the CandlePatterns class with configurations for a specific timeframe.

        Parameters:
            candles_config (dict): Dictionary containing configurations for the timeframe.
        """
        self.candles_config = candles_config
        self.pattern_name = candles_config.get('barsP_pattern', '0')
        self.pattern_candles_count = candles_config.get('barsP_pattern_count', 1)

        
        # Map pattern names to their corresponding methods
        self.pattern_mapping = {
            '0' : NoPattern, # No pattern
            'Engulf': Engulf,
            'Marubuzo': Marubuzo,
            'Out': Out,
            'In': In,
            'Ham': Ham,
            'InvHam': InvHam,
            'Kangoro_full': KangoroFull,
            'Kangoro_partial': KangoroPartial,
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

        # Initialize the appropriate candeles class based on cnadles config

        # Candle pattern
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
            CandlePattern: Candle pattern class instance.
        """
        if self.candle_pattern_name in self.pattern_mapping:
            candle_pattern_class = self.pattern_mapping[self.candle_pattern_name]
            return candle_pattern_class(self.pattern_candles_count)
        return None # Invalid pattern name

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

    def make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Make a trade decision based on the configured candle patterns.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str or None: 'buy', 'sell', 'both' or None based on the aggregated decisions.
        """
        decisions = []
        if self.candle_pattern_instance:
            decision = self.candle_pattern_instance.calculate_and_make_trade_decision(rates,tf_obj)
            if not decision:
                return None # No valid decisions from the pattern, exit early
            decisions.append(decision)

        # Handle individual candle conditions if defined
        candles = [candle for candle in [self.first_candle_instance, self.second_candle_instance, self.third_candle_instance] if candle is not None]
        for candle in candles:
            decision = candle.calculate_and_make_trade_decision(rates,tf_obj)
            if not decision:
                return None # No valid decisions from the candle, exit early
            decisions.append(decision)
        
        unique_decisions = set(decisions)
        if len(unique_decisions) == 1:
            return unique_decisions.pop()
        return None # Conflicting decisions


    def check_candle_patterns_active(self) -> bool:
        """
        Check if the configured candle pattern is active.
        Returns:
            bool: True if the pattern is active, False otherwise.
        """
        if self.pattern_name == '0' or not self.candle_pattern_instance:
            pattern_active = False
        else:
            pattern_active = True
        if self.first_candle_instance == '0' or not self.first_candle_instance:
            first_candle_active = False
        else:
            first_candle_active = True
        if self.second_candle_instance == '0' or not self.second_candle_instance:
            second_candle_active = False
        else:
            second_candle_active = True
        if self.third_candle_instance == '0' or not self.third_candle_instance:
            third_candle_active = False
        else:
            third_candle_active = True
        
        if pattern_active or first_candle_active or second_candle_active or third_candle_active:
            return True # At least one pattern is active
        return False # No active patterns

class Pattern:
    """
    Inithe the base class with the number of candles required for the pattern.

    Parameters:
        pattern_candles_count (int): Number of candles required for the pattern.
    """
    def __init__(self, pattern_candles_count: int):
        self.pattern_candles_count = pattern_candles_count
        self.trade_decision_method = None # Will be set by the child classes
    
    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        A placeholder method for calculating the pattern based on the provided rates.
        """
        raise NotImplementedError("Pattern calculation method not implemented")

    def get_trade_decision_method(self):
        """
        Return the trade decision method based on the pattern.
        This method should be implemented by the child classes.

        Returns:
            method: Trade decision method.
        """
        raise NotImplementedError("Trade decision method not implemented")

    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Calculate the pattern and make a trade decision based on the pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str or None: 'buy', 'sell', 'both' or None based on the aggregated decisions.
        """
        if self.trade_decision_method:
            return self.trade_decision_method(rates)
        return None # No trade decision method set


    def upper_wick_size(self, rates: pd.DataFrame, candle_i: int) -> float:
        """
        Calculate the upper wick size of a candle.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_condition (int): location of the candle in the dataframe.

        Returns:
            float: Upper wick size.
        """
        candle = rates.iloc[-candle_i]
        if Candle.candle_color(candle) == -1: # Red candle
            return candle['high'] - candle['open'] # Upper wick size of a red candle
        else:
            return candle['high'] - candle['close'] # Upper wick size of a green candle


    def lower_wick_size(self, rates: pd.DataFrame, candle_i: int) -> float:
        """
        Calculate the lower wick size of a candle.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_condition (int): location of the candle in the dataframe.

        Returns:
            float: Lower wick size.
        """
        candle = rates.iloc[-candle_i]
        if Candle.candle_color(candle) == -1: # Red candle
            return candle['close'] - candle['low'] # Lower wick size of a red candle
        else:
            return candle['open'] - candle['low'] # Lower wick size of a green candle


    def upper_wick_ratio(self, rates: pd.DataFrame, candle_i) -> float:
        """
        Calculate the upper wick ratio of a candle.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_i (int): location of the candle in the dataframe.

        Returns:
            float: Upper wick ratio.
        """
        size = self.upper_wick_size(rates, candle_i)
        body = self.body_size(rates, candle_i)
        return body / size if size != 0 else 1000.0

    def lower_wick_ratio(self, rates: pd.DataFrame,candle_i: int ) -> float:
        """
        Calculate the lower wick ratio of a candle.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_i (int): location of the candle in the dataframe.
        Returns:
            float: Lower wick ratio.
        """
        size = self.lower_wick_size(rates, candle_i)
        body = self.body_size(rates, candle_i)
        return body / size if size != 0 else 1000.0

    def wick_ratio(self, rates: pd.DataFrame, candle_i : int) -> float:
        """
        Calculate the overall wick ratio of a candle.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_i (int): location of the candle in the dataframe.

        Returns:
            float: Wick ratio.
        """
        upper = self.upper_wick_size(rates, candle_i)
        lower = self.lower_wick_size(rates, candle_i)
        if upper + lower == 0:
            return 10.0  # No wicks
        return self.body_size(rates, candle_i) / (upper + lower)

    # Candlestick patterns static methods:
    @staticmethod
    def hhhc(rates, i):
        return rates.iloc[-i]['close'] > rates.iloc[-i-1]['close'] and rates.iloc[-i]['high'] > rates.iloc[-i-1]['high']
    @staticmethod
    def lllc(rates, i):
        return rates.iloc[-i]['close'] < rates.iloc[-i-1]['close'] and rates.iloc[-i]['low'] < rates.iloc[-i-1]['low']



    @staticmethod
    def same_candle_count(rates: pd.DataFrame, candle_i: int) -> int:
        """
        Count the number of consecutive candles with the same color.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_condition (str): Condition specifying which candle to start counting from.

        Returns:
            int: Number of consecutive candles with the same color.
        """
        base_color = Candle.candle_color(rates.iloc[-candle_i])
        if base_color == 0:
            return 1
        count = 1
        while (candle_i + count) <= len(rates):
            current_color = Candle.candle_color(rates.iloc[-(candle_i + count)])
            if current_color == base_color:
                count += 1
            else:
                break
        return count

    @staticmethod
    def body_size(rates: pd.DataFrame, candle_i: int) -> float:
        """
        Calculate the body size of a candle.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).
            candle_condition (str): Condition specifying which candle to evaluate.

        Returns:
            float: Absolute body size.
        """
        candle = rates.iloc[-candle_i]
        return abs(candle['close'] - candle['open'])


class NoPattern(Pattern):
    """
    A placeholder class for no pattern.
    """

    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.no_trade_decision

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        return False

    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        return None

class Engulf(Pattern):
    """
    Initialize the Engulfing pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.engulf

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Engulfing pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Engulfing pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates.iloc[-1])
            previous_color = Candle.candle_color(rates.iloc[-2])
            if current_color == previous_color or current_color == 0 or previous_color == 0:
                return False

            current_open = rates.iloc[-1]['open']
            current_close = rates.iloc[-1]['close']
            previous_open = rates.iloc[-2]['open']
            previous_close = rates.iloc[-2]['close']

            if current_color == 1 and current_open <= previous_close and current_close >= previous_open:
                return True
            elif current_color == -1 and current_open >= previous_close and current_close <= previous_open:
                return True
            return False
        except IndexError:
            return False
    
    def engulf(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Engulfing pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or 'none'.
        """
        if self.calculate_pattern(rates):
            current_color = Candle.candle_color(rates.iloc[-1])
            if current_color == 1:
                return 'buy'
            elif current_color == -1:
                return 'sell'
        return None

class Marubuzo(Pattern):
    """
    Initialize the Marubuzo pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.marubuzo

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Marubuzo pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Marubuzo pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates.iloc[-1])
            body = self.body_size(rates, 1)
            wick_ratio = self.wick_ratio(rates, 1)
            if current_color == 0:
                return False
            if wick_ratio > 1.75 and max(self.upper_wick_ratio(rates, 1), self.lower_wick_ratio(rates, 1)) >= 3:
                return True
            return False
        except IndexError:
            return False

    def marubuzo(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Marubuzo pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or None.
        """
        if self.calculate_pattern(rates):
            current_color = Candle.candle_color(rates.iloc[-1])
            if current_color == 1:
                return 'buy'
            elif current_color == -1:
                return 'sell'
        return None
        
class Out(Pattern):
    """
    Initialize the Out pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.out
    
    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Out pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Out pattern is detected, False otherwise.
        """
        try:
            current = rates.iloc[-1]
            previous = rates.iloc[-2]
            if current['high'] > previous['high'] and current['low'] < previous['low']:
                return True
            return False
        except IndexError:
            return False

    def out(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Out pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both', or 'none'.
        """
        if self.calculate_pattern(rates):
            current = rates.iloc[-1]
            previous = rates.iloc[-2]
            return 'both' if current['close'] > previous['close'] else 'sell'
        return None


class In(Pattern):
    """
    Initialize the InBar pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.in_bar
    
    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for InBar pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if InBar pattern is detected, False otherwise.
        """
        try:
            current = rates.iloc[-1]
            previous = rates.iloc[-2]
            comparison = rates.iloc[-3]
            return current['high'] < previous['high'] and current['low'] > comparison['low']
        except IndexError:
            return False

    def in_bar(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for InBar pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None
    
class Ham(Pattern):
    """
    Initialize the Hammer pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.ham
    
    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Hammer pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Hammer pattern is detected, False otherwise.
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

    def ham(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Hammer pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None

class InvHam(Pattern):
    """
    Initialize the Inverted Hammer pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.invham
    
    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Inverted Hammer pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Inverted Hammer pattern is detected, False otherwise.
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

    def invham(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Inverted Hammer pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None

class KangoroFull(Pattern):
    """
    Initialize the Full Kangoro pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.kangoro_full

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Full Kangoro pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Full Kangoro pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates.iloc[-1])
            if current_color == 1:
                return Pattern.lhll(rates,1) and Pattern.hhhl(rates,1)
            elif current_color == -1:
                return Pattern.hhhl(rates,1) and Pattern.lhll(rates,1)
            return False
        except IndexError:
            return False

    def kangoro_full(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Full Kangoro pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None

class KangoroPartial(Pattern):
    """
    Initialize the Partial Kangoro pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.kangoro_partial

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Partial Kangoro pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Partial Kangoro pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates.iloc[-1])
            if current_color == 1:
                return self.lhll(rates,1)
            elif current_color == -1:
                return self.hhhl(rates,1)
            return False
        except IndexError:
            return False

    def kangoro_partial(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Partial Kangoro pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None

class Fakeout(Pattern):
    """
    Initialize the Fakeout pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.fakeout

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Fakeout pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Fakeout pattern is detected, False otherwise.
        """
        try:
            current_color = Candle.candle_color(rates.iloc[-1])
            previous_color = Candle.candle_color(rates.iloc[-2])
            if current_color == 1 and rates.iloc[-1]['close'] < rates.iloc[-2]['high']:
                return True
            elif current_color == -1 and rates.iloc[-1]['close'] > rates.iloc[-2]['low']:
                return True
            return False
        except IndexError:
            return False

    def fakeout(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Fakeout pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None
    
class SameCandleCount(Pattern): 
    """
    Initialize the Same Candle Count pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.same_candle_count_pattern
    
    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Same Candle Count pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Same Candle Count pattern is detected, False otherwise.
        """
        if self.same_candle_count(rates, 1) >= self.pattern_candles_count:
            return True
        return False

    def same_candle_count_pattern(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Same Candle Count pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None

class InsideBreakout(Pattern):
    """
    Initialize the Inside Breakout pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.inside_breakout

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        Check for Inside Breakout pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            True if Inside Breakout pattern is detected, False otherwise.
        """
        try:
            if not self.in_bar(rates):
                return False
            if rates.iloc[-2]['close'] > rates.iloc[-4]['high']:
                return True
            elif rates.iloc[-2]['close'] < rates.iloc[-4]['low']:
                return True
            return False
        except IndexError:
            return False

    def inside_breakout(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Inside Breakout pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.calculate_pattern(rates):
            return 'both'
        return None

class HHHCLLLC(Pattern):
    """
    Initialize the HHHCLLLC pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.hhhlclll

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        not required for this pattern
        """
        pass

    def hhhlclll(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for HHHCLLLC pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.hhhc(rates,1):
            return 'buy'
        if self.lllc(rates,1):
            return 'sell'
        return None

class InvHHHCLLLC(Pattern):
    """
    Initialize the Inverted HHHCLLLC pattern.
    """
    def __init__(self, pattern_candles_count: int):
        super().__init__(pattern_candles_count)
        self.trade_decision_method = self.invhhhlclll

    def calculate_pattern(self, rates: pd.DataFrame,tf_obj) -> bool:
        """
        not required for this pattern
        """
        pass

    def invhhhlclll(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Inverted HHHCLLLC pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'both' or None.
        """
        if self.hhhc(rates,1):
            return 'sell'
        if self.lhll(rates,1):
            return 'buy'
        return None

class Candle:
    """
    Base class for candle patterns.
    """
    def __init__(self):
        pass

    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        A placeholder method for calculating the candle pattern and making a trade decision.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or 'none'.
        """
        raise NotImplementedError("Candle calculation method not implemented")
    

    @staticmethod
    def candle_color(candle) -> int:
        """
        Determine the color of a specific candle based on condition.

        Parameters:
            cadnle (pd.DataFrame): 1 candle price data (OHLC).

        Returns:
            int: 1 for green, -1 for red, 0 for doji.
        """
        if candle['close'] > candle['open']:
            return 1  # Green
        elif candle['close'] < candle['open']:
            return -1  # Red
        else:
            return 0  # Doji
#


class NoCandle(Candle):
    """
    Placeholder class for no candle.
    """
    def __init__(self):
        super().__init__()

    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        return None

class SameDirectionCandle(Candle):
    """
    Initialize the Same Direction Candle class.
    """
    def __init__(self):
        super().__init__()
    
    
    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Same Direction Candle pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or 'none'.
        """
        color = Candle.candle_color(rates.iloc[-1])
        if color == 1:
            return 'buy'
        elif color == -1:
            return 'sell'
        return 'none'

class OppositeDirectionCandle(Candle):
    """
    Initialize the Opposite Direction Candle class.
    """
    def __init__(self):
        super().__init__()
    
    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Opposite Direction Candle pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: 'buy', 'sell', or 'none'.
        """
        color = Candle.candle_color(rates.iloc[-1])
        if color == 1:
            return 'sell'
        elif color == -1:
            return 'buy'
        return 'none'

class DojiCandle(Candle):
    """
    Initialize the Doji Candle class.
    """
    def __init__(self):
        super().__init__()
    
    def calculate_and_make_trade_decision(self, rates: pd.DataFrame,tf_obj) -> str:
        """
        Check for Doji Candle pattern.

        Parameters:
            rates (pd.DataFrame): Historical price data (OHLC).

        Returns:
            str: both or 'none'.
        """
        color = Candle.candle_color(rates.iloc[-1])
        if color == 0:
            return 'both'
        return 'none'



    


