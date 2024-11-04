# mt5_interface.py
"""
This module provides a unified interface for interacting with MetaTrader 5, whether in live trading or backtesting mode.
Functions:
    account_info(): Retrieves the account information.
    copy_rates(symbol, timeframe, count): Retrieves historical price data for a symbol and timeframe.
    copy_rates_from(symbol, timeframe, from_date, count): Retrieves historical price data for a symbol and timeframe starting from a specific date.
    copy_rates_from_pos(symbol, timeframe, pos, count): Retrieves historical price data for a symbol and timeframe starting from a specific position.
    order_send(request): Sends a trading order to MetaTrader 5.
    positions_get(ticket): Retrieves the open positions.
    symbol_info_tick(symbol): Retrieves the latest tick information for a symbol.
    symbol_info(symbol): Retrieves the symbol information.
    history_deals_get(from_date, to_date): Retrieves the historical deals within a date range.
    last_error(): Retrieves the last error message from MetaTrader 5.
    shutdown(): Shuts down the MetaTrader 5 client or backtesting server.
Constants:
    TIMEFRAMES (list): List of available timeframes.
    ORDER_TYPES (dict): Dictionary of available order types.
    TRADE_ACTIONS (dict): Dictionary of available trade actions.
    ORDER_TIME (dict): Dictionary of available order times.
    ORDER_FILLING (dict): Dictionary of available order filling types.
    TRADE_RETCODES (dict): Dictionary of trade return codes.
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