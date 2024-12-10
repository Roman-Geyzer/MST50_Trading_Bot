"""
This module contains helper functions and dictionaries for managing trades and orders in MetaTrader 5.

Functions:
    - calculate_lot_size: Calculate the lot size for a trade based on the risk percentage and stop-loss.
    - get_mt5_trade_type: Get the trade type (BUY or SELL) based on the trade direction.
    - get_trade_direction: Get the trade direction based on the order type.
    - calculate_sl_tp: Calculate the stop-loss and take-profit prices based on the given parameters.
    - calculate_sl: Calculate the stop-loss price based on the given parameters.
    - calculate_tp: Calculate the take-profit price based on the given parameters.
    - calculate_trail: Calculate the trailing stop price based on the given parameters.
    - check_trail_conditions: Check if the trailing stop conditions are met.
    - mt5_position_to_dict: Convert a position tuple to a dictionary.

Constants:
    - trade_dict (dict): Dictionary containing trade data for different trade directions and types.
    - sl_methods (dict): Dictionary containing stop-loss calculation methods.
    - tp_methods (dict): Dictionary containing take-profit calculation methods.
    - trail_methods (dict): Dictionary containing trailing stop calculation methods.
"""

from decimal import Decimal
from .mt5_interface import (
    TIMEFRAMES,
    ORDER_TYPES,
    TRADE_ACTIONS,
    ORDER_TIME,
    symbol_info,
    symbol_info_tick,
    account_info
)
from .constants import TRADE_DIRECTION
from .utils import print_with_info

# Helper dictionaries
trade_dict = {
    TRADE_DIRECTION.BUY: ORDER_TYPES['BUY'],
    TRADE_DIRECTION.SELL: ORDER_TYPES['SELL'],
}

# Helper functions

def calculate_lot_size(symbol: str, trade_risk_percent: float,fixed_order_size: bool, sl: float) -> float:
    """
    Calculate the lot size for a trade based on stop-loss (SL), symbol, and risk percentage.

    Parameters:
        symbol (str): Symbol name.
        trade_risk_percent (float): Risk percentage per trade.
        sl (float): Stop-loss value.
        fixed_order_size (bool): Whether to use a fixed order size.

    Returns:
        float: Calculated lot size.
    """

    # Get account balance
    # TODO - update the method to get the account balance in the header (it's the 3rd time we are calling it in the open trade loop)
    account_balance = account_info()['balance']
    if fixed_order_size:
        lot_size = trade_risk_percent * account_balance / 100000
    else:
        n_tick_value = symbol_info_tick(symbol)['bid']
        if sl == 0:
            sl = 1
        if n_tick_value == 0:
            n_tick_value = 0.00001

        # Get point size
        point_size = symbol_info(symbol)['point']

        # Calculate lot size
        lot_size = (account_balance * trade_risk_percent / 100) / (sl / point_size * n_tick_value)

    # Normalize lot size to 2 decimal places
    lot_size = round(lot_size, 2)
    lot_size = max(0.01, lot_size)  # Minimum lot size is 0.01
    return lot_size

def get_mt5_trade_type(direction):
    """
    Get the trade type (BUY or SELL) based on the trade direction.

    Parameters:
        direction (int): Trade direction.

    Returns:
        int: MT5 trade type.
    """
    return trade_dict[direction]

def get_trade_direction(trade_type):
    """
    Get the trade direction based on the order type.

    Parameters:
        trade_type (int): Order type.

    Returns:
        int: Trade direction.
    """
    if trade_type == ORDER_TYPES['BUY']:
        return TRADE_DIRECTION.BUY
    elif trade_type == ORDER_TYPES['SELL']:
        return TRADE_DIRECTION.SELL
    else:
        raise ValueError(f"Invalid trade type: {trade_type}")

def calculate_sl_tp(price, direction, sl_method, tp_method, symbol, rates):
    """
    Calculate the stop-loss and take-profit prices based on the given parameters.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_method (function): Stop-loss calculation method (sl param is prepuplated using patial).
        tp_method (function): Take-profit calculation method (tp param is prepuplated using patial).
        symbol (str): Symbol name.
        rates (np.recarray): Historical price data.

    Returns:
        tuple: (stop-loss price, take-profit price)
    """
    symbol_i = symbol_info(symbol)
    pip = symbol_i['point'] * 10

    sl = sl_method(price = price, direction = direction, symbol = symbol, pip = pip, rates = rates)
    tp = tp_method(price = price, direction = direction, symbol = symbol, pip = pip, rates = rates, sl = sl)

    # Round to the number of decimal places based on the disigt (pip size)
    sl = round(sl, symbol_i['digits'])
    tp = round(tp, symbol_i['digits'])

    return sl, tp

# SL methods
def get_sl_method(sl_method_name):
    """
    Get the SL method function based on the method name.

    Parameters:
        sl_method_name (str): SL calculation method name.

    Returns:
        function: SL method function.
    """
    return sl_methods.get(sl_method_name)

def calculate_sl(price, direction, sl_method_name, sl_param, symbol, pip, rates):
    """
    Calculate the stop loss (SL) price using sl_method based on the given parameters.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_method_name (str): SL calculation method name.
        sl_param (float): SL parameter value.
        symbol (str): Symbol name.
        pip (float): pip value for the symbol.

    Returns:
        float: Calculated SL price.
    """
    # Look up the SL method function based on the method name
    sl_method_func = sl_methods.get(sl_method_name)


    # Call the SL method function with the provided parameters
    return sl_method_func(price, direction, sl_param, symbol, pip,rates)

def UsePerc_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL using percentage.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): SL percentage.
        symbol (str): Symbol name.
        pip (float): pip value.

    Returns:
        float: Calculated SL price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        sl = price - sl_param * price / 100
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param * price / 100
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

def UseFixed_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL using a fixed number of pips.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): SL parameter.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated SL price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        sl = price - sl_param * pip
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param * pip
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

def UseCandles_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL based on candle data.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (int): Number of candles to consider.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your candle data availability
    pass

def UseSR_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL based on Support and Resistance levels.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): Parameter for SR calculation.
        symbol (str): Symbol name.
        pip (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your SR levels
    pass

def UseTrend_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL based on trend lines.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): Parameter for trend calculation.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your trend line data
    pass

def UseMA_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL based on Moving Averages.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (int): MA period.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated SL price.
    """
    ma_column = f'MA_{sl_param}'
    ma = rates[ma_column][-1]
    return ma


def UseATR_SL(price, direction, sl_param, symbol, pip,rates):
    """
    Calculate SL based on Average True Range (ATR).

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (int): MA period.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated SL price.
    """
    atr = rates['ATR'][-1]
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        sl = price - sl_param * atr
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param * atr
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

sl_methods = {
    'UsePerc_SL': UsePerc_SL,
    'UseFixed_SL': UseFixed_SL,
    'UseCandles_SL': UseCandles_SL,
    'UseSR_SL': UseSR_SL,
    'UseTrend_SL': UseTrend_SL,
    'UseMA_SL': UseMA_SL,
    'UseATR_SL': UseATR_SL,
}

# TP methods
def get_tp_method(tp_method_name):
    """
    Get the TP method function based on the method name.

    Parameters:
        tp_method_name (str): TP calculation method name.

    Returns:
        function: TP method function.
    """
    return tp_methods.get(tp_method_name)


def calculate_tp(price, direction, tp_method_name, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate the take profit (TP) price using tp_method based on the given parameters.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_method_name (str): TP calculation method name.
        tp_param (float): TP parameter value.
        symbol (str): Symbol name.
        pip (float): Pip value for the symbol.
        sl (float, optional): Stop-loss price. Required for certain TP methods.

    Returns:
        float: Calculated TP price.
    """
    # Look up the TP method function based on the method name
    tp_method_func = tp_methods.get(tp_method_name)

    # Call the TP method function with the provided parameters
    return tp_method_func(price, direction, tp_param, symbol, pip, rates, sl)


def UsePerc_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP using percentage.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): TP percentage.
        symbol (str): Symbol name.
        pip (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        tp = price + tp_param * price / 100
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * price / 100
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseFixed_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP using a fixed number of pips.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): TP parameter.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated TP price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        tp = price + tp_param * pip
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * pip
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseCandles_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP based on candle data.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (int): Number of candles to consider.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated TP price.
    """
    tp = max(rates['close'][-tp_param:-1])
    return tp

def UseSR_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP based on Support and Resistance levels.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): Parameter for SR calculation.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your SR levels
    pass

def UseTrend_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP based on trend lines.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): Parameter for trend calculation.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your trend line data
    pass

def UseMA_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP based on Moving Averages.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (int): MA period.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated TP price.
    """
    ma_column = f'MA_{tp_param}'
    ma = rates[ma_column][-1]
    return ma

def UseRR_TP(price, direction, tp_param, symbol, pip, rates=None, sl=None):
    """
    Calculate TP using Risk-Reward ratio.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): RR ratio.
        symbol (str): Symbol name.
        pip (float): Pip value.
        sl (float): Stop-loss price.

    Returns:
        float: Calculated TP price.
    """
    risk = abs(price - sl)
    reward = risk * tp_param
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        tp = price + reward
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        tp = price - reward
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseATR_TP(price, direction, tp_param, symbol, pip,rates=None, sl=None):
    """
    Calculate TP based on Average True Range (ATR).

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (int): ATR period.
        symbol (str): Symbol name.
        pip (float): Pip value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your ATR data
    atr = rates['ATR'][-1]
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        tp = price + tp_param * atr
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * atr
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

tp_methods = {
    'UsePerc_TP': UsePerc_TP,
    'UseFixed_TP': UseFixed_TP,
    'UseCandles_TP': UseCandles_TP,
    'UseSR_TP': UseSR_TP,
    'UseTrend_TP': UseTrend_TP,
    'UseMA_TP': UseMA_TP,
    'UseRR_TP': UseRR_TP,
    'UseATR_TP': UseATR_TP,
}

# Trail methods
    
def get_trail_method(trail_method_name):
    """
    Get the trail method function based on the method name.

    Parameters:
        trail_method_name (str): Trail calculation method name.

    Returns:
        function: Trail method function.
    """
    return trail_methods.get(trail_method_name)


def UsePerc_Trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop using percentage.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (float): Trail parameter value - precentage for this method.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.

    Returns:
        float : New stop-loss price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        trail_price = price - trail_param * price / 100
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        trail_price = price + trail_param * price / 100
    return trail_price

def UseFixed_Trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop using a fixed number of pips.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (float): Trail parameter value - fixed number of pips.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.

    Returns:
        float : New stop-loss price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        trail_price = price - trail_param * pip
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        trail_price = price + trail_param * pip
    return trail_price

def UseCandles_Trail_Close(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop using the close prices of the last N candles.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (int): Number of candles to consider.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.
    Returns:
        float : New stop-loss price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY: # Buy
        trail_prices = tf_rates['close'][-trail_param:-1]
        trail_price = min(trail_prices)
    elif direction == 1 or direction == TRADE_DIRECTION.SELL: # Sell
        trail_prices = tf_rates['close'][-trail_param:-1]
        trail_price = max(trail_prices)
    return trail_price

def UseCandles_Trail_Extreme(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop using the high/low prices of the last N candles.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (int): Number of candles to consider.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.
    Returns:
        float : New stop-loss price.
    """
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        trail_prices = tf_rates['low'][-trail_param:-1]
        trail_price = min(trail_prices)
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        trail_prices = tf_rates['high'][-trail_param:-1]
        trail_price = max(trail_prices)
    return trail_price

def UseSR_Trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop based on Support and Resistance levels.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (int): Number of candles to consider.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.
    Returns:
        float : New stop-loss.
    """
    # This function needs implementation based on your SR levels
    pass

def UseTrend_Trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop based on trend lines.

    Parameters:

    Returns:
        float : New stop-loss .
    """
    # This function needs implementation based on your trend line data
    pass

def UseMA_Trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop based on Moving Averages.

    Parameters:

    Returns:
        float : New stop-loss price:
    """
    ma_column = f'MA_{trail_param}'
    ma = tf_rates[ma_column][-1]
    return ma

def UseATR_Trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop based on Average True Range (ATR).

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (int): ATR multiplier.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.
    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    atr = tf_rates['ATR'][-1]
    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        trail_price = price - trail_param * atr
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        trail_price = price + trail_param * atr
    return trail_price

trail_methods = {
    'UsePerc_Trail': UsePerc_Trail,
    'UseFixed_Trail': UseFixed_Trail,
    'UseCandles_Trail_Close': UseCandles_Trail_Close,
    'UseCandles_Trail_Extreme': UseCandles_Trail_Extreme,
    'UseSR_Trail': UseSR_Trail,
    'UseTrend_Trail': UseTrend_Trail,
    'UseMA_Trail': UseMA_Trail,
    'UseATR_Trail': UseATR_Trail,
}

# Additional trailing functions




def calculate_fast_trail(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi, trail_multi, open_price= None):
    """
    Calculate trailing stop based on fast price movement on the last N minutes candles.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (int): Number of minutes to consider.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.
    Returns: float : New stop-loss (if no new SL, the current SL will be returned)
    """

    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        # Get the minimum low in the last N minutes 
        MinutesMin = m1_rates['low'][-(trail_param):].min()
        UpMove = price - MinutesMin
        if UpMove > start_multi * atr:
            trail_price = price - trail_multi * atr
            return trail_price
        else:
            return current_sl
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        # Get the maximum high in the last N minutes 
        MinutesMax = m1_rates['high'][-(trail_param):].max()
        DownMove = MinutesMax - price
        if DownMove > start_multi * atr:
            trail_price = price + trail_multi * atr
            return trail_price
        else:
            return current_sl

    

def calculate_breakeven(price, current_sl,  direction, pip, atr, tf_rates, m1_rates, trail_param,  start_multi = None, trail_multi=None, open_price= None):
    """
    Calculate trailing stop based on breakeven level.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        direction (int): Trade direction.
        pip (float): Pip value.
        atr (float): Average True Range (ATR) value.
        tf_rates (np.recarray): Historical price data for the timeframe.
        m1_rates (np.recarray): Historical price data for the M1 timeframe.
        trail_param (int): BE_ATRs - how many ATRs price move required to update SL.
        start_multi (float, optional): Start multiplier for fast trail calculation.
        trail_multi (float, optional): Trail multiplier for fast trail calculation.
        open_price (float, optional): Open price of the trade.
    Returns:
        float : New stop-loss price if conditions are met, otherwise current SL.
    """

    atr = tf_rates['ATR'][-1]

    if direction == 0 or direction == TRADE_DIRECTION.BUY:
        trail_price = open_price + pip
        if trail_price > current_sl:
            if price > open_price + trail_param * atr and price > trail_price:
                return trail_price
        return current_sl # else to both conditions
    elif direction == 1 or direction == TRADE_DIRECTION.SELL:
        trail_price = open_price - pip
        if trail_price < current_sl:
            if price < open_price - trail_param * atr and price < trail_price:
                return trail_price
        return current_sl # else to both conditions






# Additional functions as needed
#TODO: Implement this function
def mt5_position_to_dict(position):
    """
    Convert a position tuple to a dictionary.

    Parameters:
        position (mt5.Position): MT5 position object.

    Returns:
        dict: Dictionary representation of the position.
    """
    # This function needs implementation based on MT5 position object
    pass