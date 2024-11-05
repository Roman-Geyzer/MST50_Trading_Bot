# mt5_interface.py
"""
This module provides a unified interface for interacting with MetaTrader 5, whether in live trading or backtesting mode.
Functions:
    account_info: Get the account information from the server.
    copy_rates: Copy rates from the server for a symbol and timeframe.
    copy_rates_from: Copy rates from the server for a symbol, timeframe, and date.
    copy_rates_from_pos: Copy rates from the server for a symbol, timeframe, position, and count.
    order_send: Send an order request to the server.
    positions_get: Get the positions from the server.
    symbol_info_tick: Get the tick information for a symbol from the server.
    symbol_info: Get the symbol information from the server.
    symbol_select: Select a symbol on the server.
    history_deals_get: Get the history deals from the server.
    last_error: Get the last error from the server.
    shutdown: Shutdown the server.
Constants:
    TIMEFRAMES (dict): Mapping of string representations of timeframes to MetaTrader5 constants.
    ORDER_TYPES (dict): Mapping of string representations of order types to MetaTrader5 constants.
    TRADE_ACTIONS (dict): Mapping of string representations of trade actions to MetaTrader5 constants.
    ORDER_TIME (dict): Mapping of string representations of order time to MetaTrader5 constants.
    ORDER_FILLING (dict): Mapping of string representations of order filling to MetaTrader5 constants.
    TRADE_RETCODES (dict): Mapping of trade return codes to their respective descriptions.
"""

import os

# Determine the mode: live trading or backtesting
BACKTEST_MODE = os.getenv('BACKTEST_MODE', 'False') == 'True'

if BACKTEST_MODE:
    from .Backtest import mt5_backtest as mt5_module
else:
    from . import mt5_client as mt5_module

# Re-export the necessary functions and constants
# Functions
account_info = mt5_module.account_info
copy_rates = mt5_module.copy_rates
copy_rates_from = mt5_module.copy_rates_from
copy_rates_from_pos = mt5_module.copy_rates_from_pos
order_send = mt5_module.order_send
positions_get = mt5_module.positions_get
symbol_info_tick = mt5_module.symbol_info_tick
symbol_info = mt5_module.symbol_info
symbol_select = mt5_module.symbol_select
history_deals_get = mt5_module.history_deals_get
last_error = mt5_module.last_error
shutdown = mt5_module.shutdown

# Constants
TIMEFRAMES = mt5_module.TIMEFRAMES
ORDER_TYPES = mt5_module.ORDER_TYPES
TRADE_ACTIONS = mt5_module.TRADE_ACTIONS
ORDER_TIME = mt5_module.ORDER_TIME
ORDER_FILLING = mt5_module.ORDER_FILLING
TRADE_RETCODES = mt5_module.TRADE_RETCODES