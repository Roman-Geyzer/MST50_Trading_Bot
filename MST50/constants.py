# constants.py
"""
This module defines various constants and mappings used for trading with MetaTrader5.
Imports:
Constants:
    TIMEFRAME_MT5_MAPPING (dict): Mapping of string representations of timeframes to MetaTrader5 constants.
    TIMEFRAME_MAGIC_NUMBER_MAPPING (dict): Mapping of string representations of timeframes to their respective magic numbers.
    CURRENCY_MAGIC_NUMBER_MAPPING (dict): Mapping of currency pairs to their respective magic numbers.
    DEAL_TYPE (dict): Enumeration of deal types.
    DIRECTION (dict): Enumeration of trade directions.
    DEVIATION (int): Maximum deviation in points.
    SLIPPAGE (int): Maximum slippage in points.
    magic_number_base (int): Base value for magic numbers, requires currency for full number.
    performance_file (str): Filename for storing balance performance data.
"""

from .mt5_client import TIMEFRAMES, ORDER_TYPES, TRADE_ACTIONS
from enum import Enum

#Enums:
class DEAL_TYPE(Enum):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5

class TRADE_DIRECTION(Enum):
    BUY = 1
    SELL = -1
    BOTH = 0

class TRADE_TYPE(Enum):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5

class TRADE_DECISION(Enum):
    BUY = "buy"
    SELL = "sell"
    NONE = "none"

class CandleColor(Enum):
    GREEN = 'G'
    RED = 'R'
    DOJI = 'N'

class BarsTFs(Enum):
    M1 = 1
    M5 = 2
    M15 = 3
    M30 = 4
    H1 = 5
    H4 = 6
    D1 = 7
    W1 = 8



# Timeframe mapping using MT5 constants
TIMEFRAME_MT5_MAPPING = {
    'M1': TIMEFRAMES['M1'],
    'M5': TIMEFRAMES['M5'],
    'M15': TIMEFRAMES['M15'],
    'M30': TIMEFRAMES['M30'],
    'H1': TIMEFRAMES['H1'],
    'H4': TIMEFRAMES['H4'],
    'D1': TIMEFRAMES['D1'],
    'W1': TIMEFRAMES['W1']
}

# Timeframe mapping using MT5 constants
TIMEFRAME_STRING_MAPPING = {
    TIMEFRAMES['M1'] : 'M1',
    TIMEFRAMES['M5'] : 'M5',
    TIMEFRAMES['M15'] : 'M15',
    TIMEFRAMES['M30'] : 'M30',
    TIMEFRAMES['H1'] : 'H1',
    TIMEFRAMES['H4'] : 'H4',
    TIMEFRAMES['D1'] : 'D1',
    TIMEFRAMES['W1'] : 'W1'
}

# Mapping for the "timeframe" values to their respective 10_000's digit
TIMEFRAME_MAGIC_NUMBER_MAPPING = {
    'M1': 1_000_000,
    'M5': 2_000_000,
    'M15': 3_000_000,
    'M30': 4_000_000,
    'H1': 5_000_000,
    'H4': 6_000_000,
    'D1': 7_000_000,
    'W1': 8_000_000
}

# Mapping for the "symol" values to their respective magic numbers
SYMBOL_MAGIC_NUMBER_MAPPING = {
    'AUDCAD': 1,
    'AUDCHF': 2,
    'AUDJPY': 3,
    'AUDNZD': 4,
    'AUDUSD': 5,
    'CADCHF': 6,
    'CADJPY': 7,
    'CHFJPY': 8,
    'EURAUD': 9,
    'EURCAD': 10,
    'EURCHF': 11,
    'EURGBP': 12,
    'EURJPY': 13,
    'EURNZD': 14,
    'EURUSD': 15,
    'GBPAUD': 16,
    'GBPCAD': 17,
    'GBPCHF': 18,
    'GBPJPY': 19,
    'GBPNZD': 20,
    'GBPUSD': 21,
    'NZDCAD': 22,
    'NZDCHF': 23,
    'NZDJPY': 24,
    'NZDUSD': 25,
    'USDCAD': 26,
    'USDCHF': 27,
    'USDJPY': 28,
    'GOLD': 40,
    'SILVER': 41,
    'SP500': 50,
    'Oil': 60,
    'Gas': 61
}



# Other constants
DEVIATION = 10  # Max deviation in points
SLIPPAGE = 3  # Max slippage in points
magic_number_base = 50_000_000 # base of magic number, still needs the currency for full number
performance_file = 'balance_performance.csv'  # File to write balance performance data

