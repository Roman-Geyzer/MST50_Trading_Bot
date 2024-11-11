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

def calculate_lot_size(symbol: str, trade_risk_percent: float, sl: float) -> float:
    """
    Calculate the lot size for a trade based on stop-loss (SL), symbol, and risk percentage.

    Parameters:
        symbol (str): Symbol name.
        trade_risk_percent (float): Risk percentage per trade.
        sl (float): Stop-loss value.

    Returns:
        float: Calculated lot size.
    """
    # Get the symbol tick value
    n_tick_value = symbol_info_tick(symbol)['bid']
    if sl == 0:
        sl = 1
    if n_tick_value == 0:
        n_tick_value = 0.00001

    # Get account balance
    account_balance = account_info()['balance']

    # Get point size
    point_size = symbol_info(symbol)['point']

    # Calculate lot size
    lot_size = (account_balance * trade_risk_percent / 100) / (sl / point_size * n_tick_value)

    # Normalize lot size to 2 decimal places
    lot_size = round(lot_size, 2)

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

def calculate_sl_tp(price, direction, sl_method, sl_param, tp_method, tp_param, symbol):
    """
    Calculate the stop-loss and take-profit prices based on the given parameters.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_method (str): Stop-loss calculation method name.
        sl_param (float): Stop-loss parameter value.
        tp_method (str): Take-profit calculation method name.
        tp_param (float): Take-profit parameter value.
        symbol (str): Symbol name.

    Returns:
        tuple: (stop-loss price, take-profit price)
    """
    symbol_i = symbol_info(symbol)
    point = symbol_i['point']

    sl = calculate_sl(price, direction, sl_method, sl_param, symbol, point)
    if tp_method == 'UseRR_TP':
        tp = calculate_tp(price, direction, tp_method, tp_param, symbol, point, sl)
    else:
        tp = calculate_tp(price, direction, tp_method, tp_param, symbol, point)

    # Round to the number of decimal places based on the point size
    point_decimal = Decimal(str(point))
    decimal_places = -point_decimal.as_tuple().exponent
    decimal_places = min(decimal_places, 5)
    sl = round(sl, decimal_places)
    tp = round(tp, decimal_places)

    return sl, tp

# SL methods
def calculate_sl(price, direction, sl_method_name, sl_param, symbol, point):
    """
    Calculate the stop loss (SL) price using sl_method based on the given parameters.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_method_name (str): SL calculation method name.
        sl_param (float): SL parameter value.
        symbol (str): Symbol name.
        point (float): Point value for the symbol.

    Returns:
        float: Calculated SL price.
    """
    # Look up the SL method function based on the method name
    sl_method_func = sl_methods.get(sl_method_name)
    if sl_method_func is None:
        raise ValueError(f"Invalid SL method: {sl_method_name}")

    # Call the SL method function with the provided parameters
    return sl_method_func(price, direction, sl_param, symbol, point)

def UsePerc_SL(price, direction, sl_param, symbol, point):
    """
    Calculate SL using percentage.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): SL percentage.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    if direction == TRADE_DIRECTION.BUY:
        sl = price - sl_param * price / 100
    elif direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param * price / 100
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

def UseFixed_SL(price, direction, sl_param, symbol, point):
    """
    Calculate SL using a fixed number of points.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): SL parameter.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    if direction == TRADE_DIRECTION.BUY:
        sl = price - sl_param * point
    elif direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param * point
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

def UseCandles_SL(price, direction, sl_param, symbol, point):
    """
    Calculate SL based on candle data.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (int): Number of candles to consider.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your candle data availability
    pass

def UseSR_SL(price, direction, sl_param, symbol, point):
    """
    Calculate SL based on Support and Resistance levels.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): Parameter for SR calculation.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your SR levels
    pass

def UseTrend_SL(price, direction, sl_param, symbol, point):
    """
    Calculate SL based on trend lines.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (float): Parameter for trend calculation.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your trend line data
    pass

def UseMA_SL(price, direction, sl_param, symbol, point):
    """
    Calculate SL based on Moving Averages.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        sl_param (int): MA period.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated SL price.
    """
    # This function needs implementation based on your MA data
    pass

sl_methods = {
    'UsePerc_SL': UsePerc_SL,
    'UseFixed_SL': UseFixed_SL,
    'UseCandles_SL': UseCandles_SL,
    'UseSR_SL': UseSR_SL,
    'UseTrend_SL': UseTrend_SL,
    'UseMA_SL': UseMA_SL,
}

# TP methods
def calculate_tp(price, direction, tp_method_name, tp_param, symbol, point, sl=None):
    """
    Calculate the take profit (TP) price using tp_method based on the given parameters.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_method_name (str): TP calculation method name.
        tp_param (float): TP parameter value.
        symbol (str): Symbol name.
        point (float): Point value for the symbol.
        sl (float, optional): Stop-loss price. Required for certain TP methods.

    Returns:
        float: Calculated TP price.
    """
    # Look up the TP method function based on the method name
    tp_method_func = tp_methods.get(tp_method_name)
    if tp_method_func is None:
        raise ValueError(f"Invalid TP method: {tp_method_name}")

    # Call the TP method function with the provided parameters
    if tp_method_name == 'UseRR_TP' and sl is not None:
        return tp_method_func(price, direction, tp_param, symbol, point, sl)
    else:
        return tp_method_func(price, direction, tp_param, symbol, point)

def UsePerc_TP(price, direction, tp_param, symbol, point):
    """
    Calculate TP using percentage.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): TP percentage.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    if direction == TRADE_DIRECTION.BUY:
        tp = price + tp_param * price / 100
    elif direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * price / 100
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseFixed_TP(price, direction, tp_param, symbol, point):
    """
    Calculate TP using a fixed number of points.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): TP parameter.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    if direction == TRADE_DIRECTION.BUY:
        tp = price + tp_param * point
    elif direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * point
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseCandles_TP(price, direction, tp_param, symbol, point):
    """
    Calculate TP based on candle data.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (int): Number of candles to consider.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your candle data availability
    pass

def UseSR_TP(price, direction, tp_param, symbol, point):
    """
    Calculate TP based on Support and Resistance levels.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): Parameter for SR calculation.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your SR levels
    pass

def UseTrend_TP(price, direction, tp_param, symbol, point):
    """
    Calculate TP based on trend lines.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): Parameter for trend calculation.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your trend line data
    pass

def UseMA_TP(price, direction, tp_param, symbol, point):
    """
    Calculate TP based on Moving Averages.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (int): MA period.
        symbol (str): Symbol name.
        point (float): Point value.

    Returns:
        float: Calculated TP price.
    """
    # This function needs implementation based on your MA data
    pass

def UseRR_TP(price, direction, tp_param, symbol, point, sl):
    """
    Calculate TP using Risk-Reward ratio.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        tp_param (float): RR ratio.
        symbol (str): Symbol name.
        point (float): Point value.
        sl (float): Stop-loss price.

    Returns:
        float: Calculated TP price.
    """
    risk = abs(price - sl)
    reward = risk * tp_param
    if direction == TRADE_DIRECTION.BUY:
        tp = price + reward
    elif direction == TRADE_DIRECTION.SELL:
        tp = price - reward
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
}

# Trail methods
def check_trail_conditions(price, direction, trail_price, current_sl, both_sides_trail):
    """
    Check if the trailing stop conditions are met.

    Parameters:
        price (float): Current price.
        direction (int): Trade direction.
        trail_price (float): Proposed new trailing stop price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    if both_sides_trail:
        return trail_price
    if direction == 0: # Buy
        if trail_price > current_sl:
            return trail_price
        return None  # No change to SL
    elif direction == 1: # Sell
        if trail_price < current_sl:
            return trail_price
        return None
    else:
        raise ValueError(f"Invalid trade direction: {direction}")

def calculate_trail(price, current_sl, both_sides_trail, direction, trail_method_name, trail_param, symbol, point, rates_df):
    """
    Calculate the trailing stop price using trail_method based on the given parameters.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_method_name (str): Trail calculation method name.
        trail_param (float): Trail parameter value.
        symbol (str): Symbol name.
        point (float): Point value for the symbol.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    # Look up the trail method function based on the method name
    trail_method_func = trail_methods.get(trail_method_name)
    if trail_method_func is None:
        raise ValueError(f"Invalid trail method: {trail_method_name}")

    # Call the trail method function with the provided parameters
    return trail_method_func(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df)

def UsePerc_Trail(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop using percentage.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (float): Trail percentage.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    if direction == TRADE_DIRECTION.BUY:
        trail_price = price - trail_param * price / 100
    elif direction == TRADE_DIRECTION.SELL:
        trail_price = price + trail_param * price / 100
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return check_trail_conditions(price, direction, trail_price, current_sl, both_sides_trail)

def UseFixed_Trail(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop using a fixed number of points.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (float): Trail parameter.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    if direction == TRADE_DIRECTION.BUY:
        trail_price = price - trail_param * point
    elif direction == TRADE_DIRECTION.SELL:
        trail_price = price + trail_param * point
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return check_trail_conditions(price, direction, trail_price, current_sl, both_sides_trail)

def UseCandles_Trail_Close(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop using the close prices of the last N candles.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (int): Number of candles to consider.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    if direction == 0: # Buy
        trail_prices = rates_df['close'][-trail_param:-1]
        trail_price = min(trail_prices)
    elif direction == 1: # Sell
        trail_prices = rates_df['close'][-trail_param:-1]
        trail_price = max(trail_prices)
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return check_trail_conditions(price, direction, trail_price, current_sl, both_sides_trail)

def UseCandles_Trail_Extreme(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop using the high/low prices of the last N candles.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (int): Number of candles to consider.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    if direction == TRADE_DIRECTION.BUY:
        trail_prices = rates_df['low'][-trail_param:-1]
        trail_price = min(trail_prices)
    elif direction == TRADE_DIRECTION.SELL:
        trail_prices = rates_df['high'][-trail_param:-1]
        trail_price = max(trail_prices)
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return check_trail_conditions(price, direction, trail_price, current_sl, both_sides_trail)

def UseSR_Trail(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop based on Support and Resistance levels.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (float): Parameter for SR calculation.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    # This function needs implementation based on your SR levels
    pass

def UseTrend_Trail(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop based on trend lines.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (float): Parameter for trend calculation.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    # This function needs implementation based on your trend line data
    pass

def UseMA_Trail(price, current_sl, both_sides_trail, direction, trail_param, symbol, point, rates_df):
    """
    Calculate trailing stop based on Moving Averages.

    Parameters:
        price (float): Current price.
        current_sl (float): Current stop-loss price.
        both_sides_trail (bool): Whether trailing is allowed in both directions.
        direction (int): Trade direction.
        trail_param (int): MA period.
        symbol (str): Symbol name.
        point (float): Point value.
        rates_df (np.recarray): Historical price data.

    Returns:
        float or None: New stop-loss price if conditions are met, otherwise None.
    """
    # This function needs implementation based on your MA data
    pass

trail_methods = {
    'UsePerc_Trail': UsePerc_Trail,
    'UseFixed_Trail': UseFixed_Trail,
    'UseCandles_Trail_Close': UseCandles_Trail_Close,
    'UseCandles_Trail_Extreme': UseCandles_Trail_Extreme,
    'UseSR_Trail': UseSR_Trail,
    'UseTrend_Trail': UseTrend_Trail,
    'UseMA_Trail': UseMA_Trail,
}

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