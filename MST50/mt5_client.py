# MST50/mt5_client.py
"""
This module provides a client interface to the MetaTrader 5 server using Pyro5.
The client interface exposes the server functions and constants to the trading platform.
Functions:
    account_info: Get the account information from the server.
    copy_rates: Copy rates from the server for a symbol and timeframe.
    order_send: Send an order request to the server.
    positions_get: Get the positions from the server.
    symbol_info_tick: Get the tick information for a symbol from the server.
    symbol_info: Get the symbol information from the server.
    history_deals_get: Get the history deals from the server.
    copy_rates_from: Copy rates from the server for a symbol, timeframe, and date.
    copy_rates_from_pos: Copy rates from the server for a symbol, timeframe, position, and count.
    last_error: Get the last error from the server.
    symbol_select: Select a symbol on the server.
    shutdown: Shutdown the server.
"""

import Pyro5.api
import numpy as np
from datetime import datetime, timedelta


# Initialize the Pyro5 proxy
mt5_server = Pyro5.api.Proxy("PYRO:trading.platform.MT5Server@localhost:9090")

# Retrieve constants from the server
constants = mt5_server.get_constants()

server_time_hours_delta = 2

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

# Expose functions
def account_info():
    info = mt5_server.account_info()
    if info is None:
        return None
    return info  # Already a dictionary with native types

def copy_rates(symbol, timeframe, count):
    rates_list = mt5_server.copy_rates(symbol, timeframe, count)
    if rates_list is None:
        return None
    # Convert list of dictionaries back to structured NumPy array
    rates_array = np.array([tuple(d.values()) for d in rates_list], dtype=dtype)
    return rates_array

def order_send(request):
    result = mt5_server.order_send(request)
    if result is None:
        return None
    return result  # Already a dictionary with native types

def positions_get(ticket=None):
    positions_list = mt5_server.positions_get(ticket)
    if positions_list is None:
        return None
    if ticket is not None:
        return positions_list[0] # Dictionary with native types
    return positions_list  # List of dictionaries with native types

def symbol_info_tick(symbol):
    tick = mt5_server.symbol_info_tick(symbol)
    if tick is None:
        return None
    return tick  # Dictionary with native types

def symbol_info(symbol):
    info = mt5_server.symbol_info(symbol)
    if info is None:
        return None
    return info  # Dictionary with native types

def history_deals_get(from_date, to_date):
    deals_list = mt5_server.history_deals_get(from_date, to_date)
    if deals_list is None:
        return None
    return deals_list  # List of dictionaries with native types

def copy_rates_from(symbol, timeframe, from_date, count):
    rates_list = mt5_server.copy_rates_from(symbol, timeframe, from_date, count)
    if rates_list is None:
        return None
    # Convert list of dictionaries back to structured NumPy array
    rates_array = np.array([tuple(d.values()) for d in rates_list], dtype=dtype)
    return rates_array

def copy_rates_from_pos(symbol, timeframe, pos, count):
    rates_list = mt5_server.copy_rates_from_pos(symbol, timeframe, pos, count+1) # collect 1 bar extra since the last bar is probably incomplete
    if rates_list is None:
        return None
    
    # Get the timestamp of the last bar in the rates_list
    last_bar_time = rates_list[-1]['time']  # get the time of the last bar
    if check_incomplete(last_bar_time):  # expected behavior - last bar is "current incomplete bar"
        del rates_list[-1]  # remove the last bar from the list - it is incomplete (expected)
    # TODO: check if I need following lines - I think not
    # else:
    #   del rates_list[0]  # remove the first bar from the list - no new bar was made (no pip recived) and so one bar is extra

    # Convert list of dictionaries back to structured NumPy array
    rates_array = np.array([tuple(d.values()) for d in rates_list], dtype=dtype)
    return rates_array

def check_incomplete(last_bar_time):
    """
    The fucntion checks if the last bar is incomplete.
    Args:
        last_bar_time (int): The timestamp of the last bar.
    Returns:
        bool: True if the last bar is incomplete, False if the last bar is complete.
    """
    # Convert the timestamp to a datetime object
    last_bar_datetime = datetime.fromtimestamp(last_bar_time)
    
    # Get the current server datetime
    now = time_current()
    
    # TODO: update for server time
    # Compare day, hour, and minute
    if (last_bar_datetime.day == now.day and
        last_bar_datetime.hour == now.hour and
        last_bar_datetime.minute == now.minute):
        # The last bar is incomplete - this is what expected
        return True
    # The last bar is complete - unusual but still possible
    return False

def last_error():
    error = mt5_server.last_error()
    return error  # Tuple with error code and description

def symbol_select(symbol, select=True):
    return mt5_server.symbol_select(symbol, select)

def shutdown():
    mt5_server._pyroRelease()



def time_current():
    return datetime.now() + timedelta(hours=server_time_hours_delta)