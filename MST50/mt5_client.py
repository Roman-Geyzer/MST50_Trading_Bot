# trading_bot/mt5_client.py

import Pyro5.api

# Initialize the Pyro5 proxy
mt5_server = Pyro5.api.Proxy("PYRO:trading.platform.MT5Server@localhost:9090")

# Retrieve constants from the server
constants = mt5_server.get_constants()

# Expose functions
def account_info():
    return mt5_server.account_info()

def initialize_mt5():
    return mt5_server.initialize_mt5()

def login_mt5(account, password, server):
    return mt5_server.login_mt5(account, password,server)

def shutdown_mt5():
    return mt5_server.shutdown_mt5()


def copy_rates(symbol, timeframe, count):
    return mt5_server.copy_rates(symbol, timeframe, count)

def order_send(request):
    return mt5_server.order_send(request)

def positions_get(symbol):
    return mt5_server.positions_get(symbol)

def orders_get(symbol):
    return mt5_server.orders_get(symbol)

def order_modify(request):
    return mt5_server.order_modify(request)

def symbol_info_tick(symbol):
    return mt5_server.symbol_info_tick(symbol)

def symbol_info(symbol):
    return mt5_server.symbol_info(symbol)

def history_deals_get(request):
    return mt5_server.history_deals_get(request)

def copy_rates_from(symbol, timeframe, datetime_from, count):
    return mt5_server.copy_rates_from(symbol, timeframe, datetime_from, count)

def copy_rates_from_pos(symbol, timeframe, pos, count):
    return mt5_server.copy_rates_from_pos(symbol, timeframe, pos, count)

# Expose constants
TIMEFRAMES = constants['TIMEFRAMES']
ORDER_TYPES = constants['ORDER_TYPES']
TRADE_ACTIONS = constants['TRADE_ACTIONS']
ORDER_TIME = constants['ORDER_TIME']
ORDER_FILLING = constants['ORDER_FILLING']

def shutdown():
    mt5_server._pyroRelease()