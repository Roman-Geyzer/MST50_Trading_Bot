# MST50/mt5_backtest.py

import numpy as np

def get_constants():
    constants = {
        'TIMEFRAMES': {
            'M1': TIMEFRAME_M1,
            'M5': TIMEFRAME_M5,
            'M15': TIMEFRAME_M15,
            'M30': TIMEFRAME_M30,
            'H1': TIMEFRAME_H1,
            'H4': TIMEFRAME_H4,
            'D1': TIMEFRAME_D1,
            'W1': TIMEFRAME_W1,
        },
        'ORDER_TYPES': {
            'BUY': ORDER_TYPE_BUY,
            'BUY_LIMIT': ORDER_TYPE_BUY_LIMIT,
            'BUY_STOP': ORDER_TYPE_BUY_STOP,
            'BUY_STOP_LIMIT': ORDER_TYPE_BUY_STOP_LIMIT,
            'SELL': ORDER_TYPE_SELL,
            'SELL_LIMIT': ORDER_TYPE_SELL_LIMIT,
            'SELL_STOP': ORDER_TYPE_SELL_STOP,
            'SELL_STOP_LIMIT': ORDER_TYPE_SELL_STOP_LIMIT,
        },
        'TRADE_ACTIONS': {
            'DEAL': TRADE_ACTION_DEAL,
            'PENDING': TRADE_ACTION_PENDING,
            'MODIFY': TRADE_ACTION_MODIFY,
            'REMOVE': TRADE_ACTION_REMOVE,
            'CLOSE_BY': TRADE_ACTION_CLOSE_BY,
            'SLTP': TRADE_ACTION_SLTP,
            'DONE': TRADE_RETCODE_DONE
        },
        'ORDER_TIME': {
            'GTC': ORDER_TIME_GTC,
            'SPECIFIED': ORDER_TIME_SPECIFIED
        },
        'ORDER_FILLING': {
            'FOK': ORDER_FILLING_FOK
        },
        'TRADE_RETCODES' : {
            'REJECT' : TRADE_RETCODE_REJECT,
            'CANCEL' : TRADE_RETCODE_CANCEL,
            'PLACED' : TRADE_RETCODE_PLACED,
            'DONE' : TRADE_RETCODE_DONE,
            'DONE_PARTIAL' : TRADE_RETCODE_DONE_PARTIAL,
            'ERROR' : TRADE_RETCODE_ERROR,
            'TIMEOUT' : TRADE_RETCODE_TIMEOUT,
            'INVALID' : TRADE_RETCODE_INVALID,
            'INVALID_VOLUME' : TRADE_RETCODE_INVALID_VOLUME,
            'INVALID_PRICE' : TRADE_RETCODE_INVALID_PRICE,
            'INVALID_STOPS' : TRADE_RETCODE_INVALID_STOPS,
            'TRADE_DISABLED' : TRADE_RETCODE_TRADE_DISABLED,
            'MARKET_CLOSED' : TRADE_RETCODE_MARKET_CLOSED,
            'NO_MONEY' : TRADE_RETCODE_NO_MONEY,
            'PRICE_CHANGED' : TRADE_RETCODE_PRICE_CHANGED,
            'PRICE_OFF' : TRADE_RETCODE_PRICE_OFF,
            'INVALID_EXPIRATION' : TRADE_RETCODE_INVALID_EXPIRATION,
            'ORDER_CHANGED' : TRADE_RETCODE_ORDER_CHANGED,
            'TOO_MANY_REQUESTS' : TRADE_RETCODE_TOO_MANY_REQUESTS,
            'NO_CHANGES' : TRADE_RETCODE_NO_CHANGES,
            'SERVER_DISABLES_AT' : TRADE_RETCODE_SERVER_DISABLES_AT,
            'CLIENT_DISABLES_AT' : TRADE_RETCODE_CLIENT_DISABLES_AT,
            'LOCKED' : TRADE_RETCODE_LOCKED,
            'FROZEN' : TRADE_RETCODE_FROZEN,
            'INVALID_FILL' : TRADE_RETCODE_INVALID_FILL,
            'CONNECTION' : TRADE_RETCODE_CONNECTION,
            'ONLY_REAL' : TRADE_RETCODE_ONLY_REAL,
            'LIMIT_ORDERS' : TRADE_RETCODE_LIMIT_ORDERS,
            'LIMIT_VOLUME' : TRADE_RETCODE_LIMIT_VOLUME,
            'INVALID_ORDER' : TRADE_RETCODE_INVALID_ORDER,
            'POSITION_CLOSED' : TRADE_RETCODE_POSITION_CLOSED,
            'INVALID_CLOSE_VOLUME' : TRADE_RETCODE_INVALID_CLOSE_VOLUME,
            'CLOSE_ORDER_EXIST' : TRADE_RETCODE_CLOSE_ORDER_EXIST,
            'LIMIT_POSITIONS' : TRADE_RETCODE_LIMIT_POSITIONS,
            'REJECT_CANCEL' : TRADE_RETCODE_REJECT_CANCEL,
            'LONG_ONLY' : TRADE_RETCODE_LONG_ONLY,
            'SHORT_ONLY' : TRADE_RETCODE_SHORT_ONLY,
            'CLOSE_ONLY' : TRADE_RETCODE_CLOSE_ONLY,
            'FIFO_CLOSE' : TRADE_RETCODE_FIFO_CLOSE
        },
    }
    # Convert NumPy types to native Python types
    return self._convert_numpy_types(constants)

constants = get_constants()

# Expose constants
TIMEFRAMES = constants['TIMEFRAMES']
ORDER_TYPES = constants['ORDER_TYPES']
TRADE_ACTIONS = constants['TRADE_ACTIONS']
ORDER_TIME = constants['ORDER_TIME']
ORDER_FILLING = constants['ORDER_FILLING']
TRADE_RETCODES = constants['TRADE_RETCODES']

# Define the data types for the structured array
dtype = [
    ('time', 'int64'),
    ('open', 'float64'),
    ('high', 'float64'),
    ('low', 'float64'),
    ('close', 'float64'),
    ('tick_volume', 'int64'),
    ('spread', 'int64'),
    ('real_volume', 'int64')
]

#TODO: Implement the following functions - mimic the behavior of the MT5 server
def account_info():
    pass

def copy_rates(symbol, timeframe, count):
    pass


def order_send(request):
    pass

def positions_get(ticket):
    pass

def symbol_info_tick(symbol):
    pass

def symbol_info(symbol):
    pass

def history_deals_get(from_date, to_date):
    # do I need this method in backtes?
    pass

def copy_rates_from(symbol, timeframe, from_date, count):
    pass

def copy_rates_from_pos(symbol, timeframe, pos, count):
    pass

def last_error():
    pass

def symbol_select(symbol):
    pass

def shutdown():
    pass

# Helper function to convert NumPy types to native Python types
# do I need this method in backtest?
def _convert_numpy_types(constants):
    for key, value in constants.items():
        if isinstance(value, dict):
            constants[key] = _convert_numpy_types(value)
        elif isinstance(value, np.dtype):
            constants[key] = value.type
    return constants