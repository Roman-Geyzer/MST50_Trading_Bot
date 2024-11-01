# orders.py
"""
This module contains helper functions and dictionaries for managing trades and orders in MetaTrader 5.
Functions:
    calculate_lot_size(): Calculate the lot size for a trade based on stop-loss (SL), symbol, and risk percentage.
    calculate_sl_tp(): Calculate the stop-loss (SL) and take-profit (TP) prices based on the given parameters.
    calculate_sl(): Calculate the stop-loss (SL) price using sl_method based on the given parameters.
    calculate_tp(): Calculate the take-profit (TP) price using tp_method based on the given parameters.
    calculate_trail(): Calculate the trailing stop price using trail_method based on the given parameters.
    UsePerc_SL(): Calculate the stop loss (SL) price based on a percentage of the current price.
    UseFixed_SL(): Calculate the stop loss (SL) price based on a fixed value.
    UsePerc_TP(): Calculate the take profit (TP) price based on a percentage of the current price.
    UseFixed_TP(): Calculate the take profit (TP) price based on a fixed value.
    UseRR_TP(): Calculate the take profit (TP) price based on a risk-reward ratio.
    UsePerc_Trail(): Calculate the trailing stop price based on a percentage of the current price.
    UseFixed_Trail(): Calculate the trailing stop price based on a fixed value.
dicts:
    trade_dict (dict): Dictionary containing trade types and actions for buying and selling.
    sl_methods (dict): Dictionary containing stop-loss calculation methods.
    tp_methods (dict): Dictionary containing take-profit calculation methods.
    trail_methods (dict): Dictionary containing trailing stop calculation methods.


"""

# helper dictionarries:

from decimal import Decimal
from .mt5_interface import (TIMEFRAMES, ORDER_TYPES, TRADE_ACTIONS, ORDER_TIME,
                        symbol_info, symbol_info_tick, account_info)
from .constants import TRADE_DIRECTION
import time
from .utils import print_with_info






#TODO: check if directions are correct - buy is 1 and sell is 0 ?
#TODO: update the trade_dict with the correct values
trade_dict = {
    TRADE_DIRECTION.BUY: {
        'market': {
            'action': TRADE_ACTIONS['DEAL'],
            'type': ORDER_TYPES['BUY'] ,
        },
        'limit': {
            'action': TRADE_ACTIONS['PENDING'],
            'type': ORDER_TYPES['BUY_LIMIT'],

        },
        'stop': {
            'action': TRADE_ACTIONS['PENDING'],
            'type': ORDER_TYPES['BUY_STOP'],
        },
        'limit_stop': {
            'action': TRADE_ACTIONS['PENDING'],
            'type': ORDER_TYPES['BUY_STOP_LIMIT'],
        },
    },
    TRADE_DIRECTION.SELL: {
        'market': {
            'action': TRADE_ACTIONS['DEAL'],
            'type': ORDER_TYPES['SELL'],
        },
        'limit': {
            'action': TRADE_ACTIONS['PENDING'],
            'type': ORDER_TYPES['SELL_LIMIT'],
        },
        'stop': {
            'action': TRADE_ACTIONS['PENDING'],
            'type': ORDER_TYPES['SELL_STOP'],
        },
        'limit_stop': {
            'action': TRADE_ACTIONS['PENDING'],
            'type': ORDER_TYPES['SELL_STOP_LIMIT'],
        },
    },
}

# helper functions:

def calculate_lot_size(symbol, trade_risk_percent, sl):
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

    # Point() in MQL5 refers to the tick size, so we'll use symbol_info().point to get the point size
    point_size = symbol_info(symbol)['point']

    # Calculate lot size
    lot_size = (account_balance * trade_risk_percent / 100) / (sl / point_size * n_tick_value)
    
    # Normalize lot size to 2 decimal places
    lot_size = round(lot_size, 2)
    
    return lot_size

def get_mt5_trade_data(direction, trade_type):
    """
    function to get the trade data from the trade_dict
    input: direction, trade_type
    output: trade_data
    """
    print(f"direction: {direction}, trade_type: {trade_type}")
    trade_data = trade_dict[direction][trade_type]
    return trade_data

def get_trade_direction(trade_type):
    if trade_type == ORDER_TYPES['BUY'] or trade_type == ORDER_TYPES['BUY_LIMIT'] or trade_type == ORDER_TYPES['BUY_STOP'] or trade_type == ORDER_TYPES['BUY_STOP_LIMIT']:
        return TRADE_DIRECTION.BUY
    elif trade_type == ORDER_TYPES['SELL'] or trade_type == ORDER_TYPES['SELL_LIMIT'] or trade_type == ORDER_TYPES['SELL_STOP'] or trade_type == ORDER_TYPES['SELL_STOP_LIMIT']:
        return TRADE_DIRECTION.SELL
    else:
        raise ValueError(f"Invalid trade type: {trade_type}")

def calculate_sl_tp(price, direction,sl_method, sl_param, tp_method, tp_param, symbol):
    symbol_i = symbol_info(symbol)
    point = symbol_i['point']

    sl = calculate_sl(price, direction, sl_method, sl_param, symbol, point)
    if tp_method == 'UseRR_TP':
        tp = calculate_tp(price, direction, tp_method, tp_param, symbol, point, sl)
    else:
        tp = calculate_tp(price, direction, tp_method, tp_param, symbol, point)

    # Round to 5 decimal places or less
    point_decimal = Decimal(str(point))
    decimal_places = -point_decimal.as_tuple().exponent
    decimal_places = min(decimal_places, 5)
    sl = round(sl, decimal_places)
    tp = round(tp, decimal_places)

    return sl, tp
#obsolete - since use of
"""
def mt5_position_to_dict(position):
    recives a position object (tuple) and returns a dictionary with the position data
    if isinstance(position, dict):
        return position
    position = position[0]
    position_dict = {
        'ticket': position[0],
        'time': position[1],
        'time_msc': position[2],
        'time_update': position[3],
        'time_update_msc': position[4],
        'type': position[5],
        'magic': position[6],
        'identifier': position[7],
        'reason': position[8],
        'volume': position[9],
        'price_open': position[10],
        'sl': position[11],
        'tp': position[12],
        'price_current': position[13],
        'swap': position[14],
        'profit': position[15],
        'symbol': position[16],
        'comment': position[17],
        'external_id': position[18],
    }
    return position_dict
#
"""

# SL methods:
def calculate_sl(price, direction, sl_method_name, sl_param, symbol, point):
    """
    Calculate the stop loss (SL) price using sl_method based on the given parameters.
    Args:
        price (float): Current price.
        direction (str): Trade direction ('buy' or 'sell').
        sl_method_name (str): SL calculation method name.
        sl_param (float): SL parameter value.
        symbol (str): Symbol name.
        point (float): Point value for the symbol.
    Returns:
        float: Calculated SL price.
    """
    # Look up the SL method function based on the method name
    sl_method = sl_methods.get(sl_method_name)
    if sl_method is None:
        raise ValueError(f"Invalid SL method: {sl_method_name}")

    # Call the SL method function with the provided parameters
    return sl_method(price, direction, sl_param, symbol, point)


def UsePerc_SL(price, direction, sl_param, symbol, point):
    if direction == TRADE_DIRECTION.BUY: 
        sl = price - sl_param 
    elif direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param 
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

def UseCandels_SL(price, direction, sl_param, symbol, point):
    pass

def UseSR_SL(price, direction, sl_param, symbol, point):
    pass

def UseTrend_SL(price, direction, sl_param, symbol, point):
    pass

def UseMA_SL(price, direction, sl_param, symbol, point):
    pass

def UseFixed_SL(price, direction, sl_param, symbol, point):
    if direction == TRADE_DIRECTION.BUY: 
        sl = price - sl_param * point * 10
    elif direction == TRADE_DIRECTION.SELL:
        sl = price + sl_param * point * 10
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return sl

sl_methods = {
    'UsePerc_SL': UsePerc_SL,
    'UseCandels_SL': UseCandels_SL,
    'UseSR_SL': UseSR_SL,
    'UseTrend_SL': UseTrend_SL,
    'UseMA_SL': UseMA_SL,
    'UseFixed_SL': UseFixed_SL,
}

# TP methods:

def calculate_tp(price, direction, tp_method, tp_param, symbol, point):
    """
    Calculate the take profit (TP) price using tp_method based on the given parameters.
    Args:
        price (float): Current price.
        direction (str): Trade direction ('buy' or 'sell').
        tp_method (str): TP calculation method.
        tp_param (float): TP parameter value.
        symbol (str): Symbol name.
        point (float): Point value for the symbol.
    Returns:
        float: Calculated TP price.
    """
    # Look up the TP method function based on the method name
    tp_method = tp_methods.get(tp_method)
    if tp_method is None:
        raise ValueError(f"Invalid TP method: {tp_method}")
    
    # Call the TP method function with the provided parameters
    return tp_method(price, direction, tp_param, symbol, point)

def UsePerc_TP(price, direction, tp_param, symbol, point):
    if direction == TRADE_DIRECTION.BUY: 
        tp = price + tp_param * point 
    elif direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * point 
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseCandels_TP(price, direction, tp_param, symbol, point):
    pass


def UseSR_TP(price, direction, tp_param, symbol, point):
    pass


def UseTrend_TP(price, direction, tp_param, symbol, point):
    pass


def UseMA_TP(price, direction, tp_param, symbol, point):
    pass


def UseFixed_TP(price, direction, tp_param, symbol, point):
    if direction == TRADE_DIRECTION.BUY: 
        tp = price + tp_param * point * 10
    elif direction == TRADE_DIRECTION.SELL:
        tp = price - tp_param * point * 10
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return tp

def UseRR_TP(price, direction, tp_param, symbol, point, SL):
    if direction == TRADE_DIRECTION.BUY: 
        tp = price + tp_param * point
    else:
        tp = price - tp_param * point
    return tp

tp_methods = {
    'UsePerc_TP': UsePerc_TP,
    'UseCandels_TP': UseCandels_TP,
    'UseSR_TP': UseSR_TP,
    'UseTrend_TP': UseTrend_TP,
    'UseMA_TP': UseMA_TP,
    'UseFixed_TP': UseFixed_TP,
    'UseRR_TP': UseRR_TP,
}

# Trail methods:

def check_trail_conditions(price, direction, trail_price,current_sl, both_sides_trail):
    if both_sides_trail:
        return trail_price
    if direction % 2 == 0: # BUY -> trade dircetion's are 0, 2, 4, 6 
        if trail_price > current_sl:
            return trail_price 
        return None # no change to SL
    else: # SELL traded dircetion's are 1, 3, 5, 7
        if trail_price < current_sl:
            return trail_price
        return None

def calculate_trail(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    """
    Calculate the trailing stop price using trail_method based on the given parameters.
    Args:
        price (float): Current price.
        direction (str): Trade direction ('buy' or 'sell').
        trail_method (str): Trail calculation method.
        trail_param (float): Trail parameter value.
        symbol (str): Symbol name.
        point (float): Point value for the symbol.
    Returns:
        float: Calculated trailing stop price.
    """
    # Look up the trail method function based on the method name
    trail_method = trail_methods.get(trail_method)
    if trail_method is None:
        raise ValueError(f"Invalid trail method: {trail_method}")
    
    #TODO: this need more work, maybe send the position object to the trail method instead of price and direction
    # Call the trail method function with the provided parameters
    return trail_method(price,current_sl,both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df)

def UsePerc_Trail(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    if direction == TRADE_DIRECTION.BUY: 
        trail = price - trail_param * point
    elif direction == TRADE_DIRECTION.SELL:
        trail = price + trail_param * point
    else:
        raise ValueError(f"Invalid trade direction: {direction}")
    return trail

def UseCandels_Trail_Close(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    if direction % 2 == 0: # BUY -> trade dircetion's are 0, 2, 4, 6 
        trail_price = min(rates_df.iloc[-trail_param:-1]['close'])
    else: # SELL traded dircetion's are 1, 3, 5, 7
        trail_price = max(rates_df.iloc[-trail_param:-1]['close'])
    print_with_info(f"trail_price: {trail_price} , for position: {direction}")
    return check_trail_conditions(price, direction, trail_price, current_sl, both_sides_trail)

def UseCandels_Trail_Extreme(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    pass

def UseSR_Trail(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    pass

def UseTrend_Trail(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    pass

def UseMA_Trail(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    pass

def UseFixed_Trail(price, current_sl, both_sides_trail, direction, trail_method, trail_param, symbol, point, rates_df):
    trail_price = price - trail_param * point
    return check_trail_conditions(price, direction, trail_price,current_sl, both_sides_trail)

trail_methods = {
    'UsePerc_Trail': UsePerc_Trail,
    'UseCandels_Trail_Close': UseCandels_Trail_Close,
    'UseCandels_Trail_Extreme': UseCandels_Trail_Extreme,
    'UseSR_Trail': UseSR_Trail,
    'UseTrend_Trail': UseTrend_Trail,
    'UseMA_Trail': UseMA_Trail,
    'UseFixed_Trail': UseFixed_Trail,
}
