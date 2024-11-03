# mt5_backtest.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import copy
import plotly.graph_objects as go



# days set to end backtest - x days ago from today
backtest_end_relative_to_today = 180
# ----------------------------
# Constants Definition
# ----------------------------

# Timeframes
TIMEFRAME_M1 = 1
TIMEFRAME_M2 = 2
TIMEFRAME_M3 = 3
TIMEFRAME_M4 = 4
TIMEFRAME_M5 = 5
TIMEFRAME_M6 = 6
TIMEFRAME_M10 = 10
TIMEFRAME_M12 = 12
TIMEFRAME_M15 = 15
TIMEFRAME_M20 = 20
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 1 | 0x4000
TIMEFRAME_H2 = 2 | 0x4000
TIMEFRAME_H3 = 3 | 0x4000
TIMEFRAME_H4 = 4 | 0x4000
TIMEFRAME_H6 = 6 | 0x4000
TIMEFRAME_H8 = 8 | 0x4000
TIMEFRAME_H12 = 12 | 0x4000
TIMEFRAME_D1 = 24 | 0x4000
TIMEFRAME_W1 = 1 | 0x8000
TIMEFRAME_MN1 = 1 | 0xC000

# Map of timeframe constants to names
TIMEFRAMES = {
    'M1': TIMEFRAME_M1,
    'M2': TIMEFRAME_M2,
    'M3': TIMEFRAME_M3,
    'M4': TIMEFRAME_M4,
    'M5': TIMEFRAME_M5,
    'M6': TIMEFRAME_M6,
    'M10': TIMEFRAME_M10,
    'M12': TIMEFRAME_M12,
    'M15': TIMEFRAME_M15,
    'M20': TIMEFRAME_M20,
    'M30': TIMEFRAME_M30,
    'H1': TIMEFRAME_H1,
    'H2': TIMEFRAME_H2,
    'H3': TIMEFRAME_H3,
    'H4': TIMEFRAME_H4,
    'H6': TIMEFRAME_H6,
    'H8': TIMEFRAME_H8,
    'H12': TIMEFRAME_H12,
    'D1': TIMEFRAME_D1,
    'W1': TIMEFRAME_W1,
    'MN1': TIMEFRAME_MN1,
}

# Reverse map of timeframe values to names
TIMEFRAMES_REVERSE = {v: k for k, v in TIMEFRAMES.items()}

# Position types
POSITION_TYPE_BUY = 0      # Buy
POSITION_TYPE_SELL = 1     # Sell

# Position reasons
POSITION_REASON_CLIENT = 0      # Order from desktop terminal
POSITION_REASON_MOBILE = 1      # Order from mobile application
POSITION_REASON_WEB = 2         # Order from web platform
POSITION_REASON_EXPERT = 3      # Order from MQL5 program

# Order types
ORDER_TYPE_BUY = 0                      # Market Buy order
ORDER_TYPE_SELL = 1                     # Market Sell order
ORDER_TYPE_BUY_LIMIT = 2                # Buy Limit pending order
ORDER_TYPE_SELL_LIMIT = 3               # Sell Limit pending order
ORDER_TYPE_BUY_STOP = 4                 # Buy Stop pending order
ORDER_TYPE_SELL_STOP = 5                # Sell Stop pending order
ORDER_TYPE_BUY_STOP_LIMIT = 6           # Buy Stop Limit pending order
ORDER_TYPE_SELL_STOP_LIMIT = 7          # Sell Stop Limit pending order
ORDER_TYPE_CLOSE_BY = 8                 # Close a position by an opposite one

# Trade request actions
TRADE_ACTION_DEAL = 1                   # Place a market order
TRADE_ACTION_PENDING = 5                # Place a pending order
TRADE_ACTION_SLTP = 6                   # Modify SL/TP
TRADE_ACTION_MODIFY = 7                 # Modify an order
TRADE_ACTION_REMOVE = 8                 # Delete a pending order
TRADE_ACTION_CLOSE_BY = 10              # Close a position by an opposite one

# Trade return codes
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_ERROR = 10011

# Function error codes
RES_S_OK = 1           # Generic success
RES_E_FAIL = -1        # Generic fail
RES_E_INVALID_PARAMS = -2  # Invalid arguments/parameters
RES_E_NO_MEMORY = -3       # No memory condition
RES_E_NOT_FOUND = -4       # No history
RES_E_INVALID_VERSION = -5 # Invalid version
RES_E_AUTH_FAILED = -6     # Authorization failed
RES_E_UNSUPPORTED = -7     # Unsupported method
RES_E_AUTO_TRADING_DISABLED = -8  # Auto-trading disabled
RES_E_INTERNAL_FAIL = -10000      # Internal IPC general error



# Helper function

def timeframe_to_timedelta(timeframe_str):
    """
    Convert timeframe string to timedelta.
    """
    if timeframe_str.startswith('M'):
        minutes = int(timeframe_str[1:])
        return timedelta(minutes=minutes)
    elif timeframe_str.startswith('H'):
        hours = int(timeframe_str[1:])
        return timedelta(hours=hours)
    elif timeframe_str.startswith('D'):
        days = int(timeframe_str[1:])
        return timedelta(days=days)
    elif timeframe_str.startswith('W'):
        weeks = int(timeframe_str[1:])
        return timedelta(weeks=weeks)
    elif timeframe_str.startswith('MN'):
        months = int(timeframe_str[2:])
        return timedelta(days=30 * months)  # Approximate month as 30 days
    else:
        raise ValueError(f"Unknown timeframe format: {timeframe_str}")
    

# TODO: brutally coppied from utils and constants, need to think of a better way
TIMEFRAME_MT5_MAPPING_COPY = {
    'M1': TIMEFRAMES['M1'],
    'M5': TIMEFRAMES['M5'],
    'M15': TIMEFRAMES['M15'],
    'M30': TIMEFRAMES['M30'],
    'H1': TIMEFRAMES['H1'],
    'H4': TIMEFRAMES['H4'],
    'D1': TIMEFRAMES['D1'],
    'W1': TIMEFRAMES['W1']
}


def get_timeframe_string(timeframe):
    # Return the corresponding index for the given timeframe
    return TIMEFRAME_MT5_MAPPING_COPY.get(timeframe, None)

# ----------------------------
# MT5Backtest Class Definition
# ----------------------------


drive = "x:"

class MT5Backtest:
    """
    A class to simulate the MetaTrader 5 (MT5) client for backtesting purposes.
    """

    def __init__(self, strategies, data_dir="historical_data", output_dir="Backtests"):
        """
        Initialize the backtest environment.

        Parameters:
            strategies (dict): Dictionary of strategy instances.
            data_dir (str): Directory where historical CSV data is stored.
            output_dir (str): Directory where backtest outputs will be saved.
        """
        self.strategies = strategies  # Store the strategies
        self.data_dir = os.path.join(drive, data_dir)
        self.symbols_data = {}  # {symbol: {timeframe: DataFrame}}

        # Extract symbols, timeframes, and backtest parameters from strategies
        self.extract_backtest_parameters()

        self.load_data()

        # Initialize simulation parameters
        self.current_time = self.start_time
        self.end_time = self.end_time

        # Initialize account
        self.account = {
            'balance': 100000.0,
            'equity': 100000.0,
            'margin': 0.0,
            'free_margin': 100000.0,
            'profit': 0.0
        }

        # Initialize positions
        self.open_positions = {}   # {ticket: position_info}
        self.closed_positions = [] # List of closed position_info

        # Error simulation
        self.last_error_code = RES_S_OK
        self.last_error_description = "No error"

        # Ticket counter
        self.next_ticket = 1000  # Starting ticket number

        # Order management
        self.pending_orders = []  # List of pending order requests

        # Output directory setup
        self.output_dir = os.path.join(drive, output_dir)
        start_time_str = self.current_time.strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(self.output_dir, f"Backtest_{start_time_str}")
        os.makedirs(self.run_folder, exist_ok=True)

        # Initialize trade logs and account documentation
        self.trade_logs = []      # List to store trade dictionaries
        self.account_docs = []    # List to store account documentation dictionaries

        # Initialize previous hour to track hourly logging
        self.previous_hour = self.current_time.hour

    def extract_backtest_parameters(self):
        """
        Extract symbols, timeframes, start_time, end_time, and time_step from strategies.
        """
        symbols_set = set()
        timeframes_set = set()
        backtest_start_dates = []
        backtest_time_steps = []

        for strategy in self.strategies.values():
            symbols_set.update(strategy.symbols)
            # Add the strategy's main timeframe
            timeframes_set.add(strategy.str_timeframe)
            # Add higher and lower timeframes from strategy
            timeframes_set.add(get_timeframe_string(strategy.higher_timeframe))
            timeframes_set.add(get_timeframe_string(strategy.lower_timeframe))
            # Collect backtest start date and time step
            backtest_start_dates.append(strategy.backtest_start_date)
            backtest_time_steps.append(strategy.backtest_tf)

        self.symbols = list(symbols_set)
        self.timeframes = list(timeframes_set)

        # Use the earliest start date among strategies
        start_time_str = min(backtest_start_dates)
        # Convert start_time_str to datetime object
        self.start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')  # Adjust format as needed

        # Set end date as 6 months before today
        self.end_time = datetime.now() - timedelta(days=backtest_end_relative_to_today)

        # Use the backtest time step (candle advance) from the strategies
        # Assuming all strategies have the same backtest time frame
        backtest_timeframe = backtest_time_steps[0]  # Assuming all are the same
        self.advance_timeframe = backtest_timeframe  # Store the timeframe string

        # Map the timeframe string to a timedelta
        self.time_step = timeframe_to_timedelta(backtest_timeframe)


    def load_data(self):
        """
        Load historical data from CSV files into symbols_data.
        Only loads data for specified symbols and timeframes.
        """
        if not self.symbols or not self.timeframes:
            print("No symbols or timeframes specified for data loading.")
            return

        for filename in os.listdir(self.data_dir):
            filepath = os.path.join(self.data_dir, filename)
            try:
                symbol, tf_name = filename.replace('.csv', '').split('_')
            except ValueError:
                print(f"Filename {filename} does not match the required format 'Symbol_Timeframe.csv'. Skipping.")
                continue

            # Only load data for specified symbols and timeframes
            if symbol not in self.symbols or tf_name not in self.timeframes:
                continue  # Skip unnecessary data

            df = pd.read_csv(filepath, parse_dates=['time'])
            df.sort_values('time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            if symbol not in self.symbols_data:
                self.symbols_data[symbol] = {}
        self.symbols_data[symbol][tf_name] = df


    def get_timeframe_name(self, timeframe):
        """
        Get the timeframe name from its value.

        Parameters:
            timeframe (int): The timeframe constant.

        Returns:
            str: The name of the timeframe.
        """
        return TIMEFRAMES_REVERSE.get(timeframe, None)

    def copy_rates(self, symbol, timeframe, count):
        """
        Simulate MT5.copy_rates() function.

        Parameters:
            symbol (str): The trading symbol.
            timeframe (int): The timeframe constant.
            count (int): Number of bars to copy.

        Returns:
            np.recarray or None: Array of rates up to current_time or None if error.
        """
        tf_name = self.get_timeframe_name(timeframe)
        if not tf_name or symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        df_up_to_now = df[df['time'] <= self.current_time]
        if df_up_to_now.empty:
            self.set_last_error(RES_E_NOT_FOUND, f"No data available up to current time for {symbol}, {tf_name}")
            return None

        rates = df_up_to_now.tail(count)
        return rates.to_records(index=False)

    def copy_rates_from_pos(self, symbol, timeframe, pos, count):
        """
        Simulate MT5.copy_rates_from_pos() function.

        Parameters:
            symbol (str): The trading symbol.
            timeframe (int): The timeframe constant.
            pos (int): Starting position.
            count (int): Number of bars to copy.

        Returns:
            np.recarray or None: Array of rates or None if error.
        """
        tf_name = self.get_timeframe_name(timeframe)
        if not tf_name or symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        if pos < 0 or pos >= len(df):
            self.set_last_error(RES_E_INVALID_PARAMS, f"Invalid position: {pos} for {symbol}, {tf_name}")
            return None

        rates = df.iloc[pos: pos + count]
        return rates.to_records(index=False)

    def copy_rates_from(self, symbol, timeframe, datetime_from, count):
        """
        Simulate MT5.copy_rates_from() function.

        Parameters:
            symbol (str): The trading symbol.
            timeframe (int): The timeframe constant.
            datetime_from (datetime): Start datetime.
            count (int): Number of bars to copy.

        Returns:
            np.recarray or None: Array of rates or None if error.
        """
        tf_name = self.get_timeframe_name(timeframe)
        if not tf_name or symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        df_from = df[df['time'] >= datetime_from]
        if df_from.empty:
            self.set_last_error(RES_E_NOT_FOUND, f"No data available from {datetime_from} for {symbol}, {tf_name}")
            return None

        rates = df_from.head(count)
        return rates.to_records(index=False)

    def account_info(self):
        """
        Simulate MT5.account_info() function.

        Returns:
            dict: Account information.
        """
        return copy.deepcopy(self.account)

    def positions_get(self, ticket=None):
        """
        Simulate MT5.positions_get() function.

        Parameters:
            ticket (int, optional): Specific ticket number to retrieve.

        Returns:
            dict or list or None: Single position dict, list of positions, or None if not found.
        """
        if ticket is not None:
            return copy.deepcopy(self.open_positions.get(ticket, None))
        else:
            return copy.deepcopy(list(self.open_positions.values())) if self.open_positions else None

    def symbol_info_tick(self, symbol):
        """
        Simulate MT5.symbol_info_tick() function.

        Parameters:
            symbol (str): The trading symbol.

        Returns:
            dict or None: Tick information or None if error.
        """
        tf_name = 'M1'  # Assuming M1 has the latest data
        if symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        current_bar = df[df['time'] <= self.current_time].tail(1)
        if current_bar.empty:
            self.set_last_error(RES_E_NOT_FOUND, f"No tick data available for {symbol} up to current time.")
            return None

        tick = {
            'time': int(current_bar['time'].iloc[0].timestamp()),
            'bid': current_bar['close'].iloc[0],  # Simplified: using close price as bid
            'ask': current_bar['close'].iloc[0] + (current_bar['spread'].iloc[0] * 0.0001),  # Simplified spread
            'last': current_bar['close'].iloc[0]
        }
        return tick

    def symbol_info(self, symbol):
        """
        Simulate MT5.symbol_info() function.

        Parameters:
            symbol (str): The trading symbol.

        Returns:
            dict: Symbol information.
        """
        # Simplified symbol information
        info = {
            'name': symbol,
            'path': symbol,
            'description': symbol,
            'digits': 5,
            'spread': 2,
            'trade_mode': 0,
            'volume_min': 0.01,
            'volume_max': 1000.0,
            'volume_step': 0.01,
            'price_min': 0.0,
            'price_max': 100000.0,
            'price_step': 0.0001,
            'visible': True,
            'enabled': True,
            'calc_mode': 0,
            'trade_mode': 0,
            'swap_mode': 0,
            'order_gtc_mode': 0,
            'option_right': 0,
            'option_mode': 0,
            'margin_mode': 0,
            'trade_execution': 0,
            'chart_mode': 0,
        }
        return info

    def symbol_select(self, symbol, select=True):
        """
        Simulate MT5.symbol_select() function.

        Parameters:
            symbol (str): The trading symbol.
            select (bool): Whether to select or deselect the symbol.

        Returns:
            bool: Success status.
        """
        # For simplicity, assume all symbols are always selected
        return True

    def history_deals_get(self, from_date, to_date):
        """
        Simulate MT5.history_deals_get() function.

        Parameters:
            from_date (datetime): Start datetime.
            to_date (datetime): End datetime.

        Returns:
            list or None: List of deal dictionaries or None if no deals found.
        """
        deals = []
        for pos in self.closed_positions:
            close_time = pos.get('close_datetime')
            if close_time and from_date <= close_time <= to_date:
                deals.append(copy.deepcopy(pos))
        return deals if deals else None

    def order_send(self, request):
        """
        Simulate MT5.order_send() function.

        Parameters:
            request (dict): Order request parameters.

        Returns:
            dict or None: Order execution result or None if error.
        """
        action = request.get('action')
        symbol = request.get('symbol')
        volume = request.get('volume', 0.01)
        order_type = request.get('type')
        price = request.get('price', 0.0)
        deviation = request.get('deviation', 10)
        comment = request.get('comment', '')
        sl = request.get('sl', 0.0)
        tp = request.get('tp', 0.0)
        magic = request.get('magic', 0)

        # Simplified order execution
        if action == TRADE_ACTION_DEAL:
            # Market order
            result = self.execute_market_order(symbol, order_type, volume, price, comment, sl, tp, magic)
            return result
        elif action == TRADE_ACTION_PENDING:
            # Pending order (limit or stop)
            result = self.execute_pending_order(symbol, order_type, volume, price, comment, sl, tp, magic)
            return result
        elif action == TRADE_ACTION_MODIFY:
            # Modify order
            result = self.modify_order(request)
            return result
        elif action == TRADE_ACTION_REMOVE:
            # Remove pending order
            result = self.remove_pending_order(request)
            return result
        elif action == TRADE_ACTION_CLOSE_BY:
            # Close by opposite position
            result = self.close_by_order(request)
            return result
        else:
            self.set_last_error(RES_E_INVALID_PARAMS, f"Unknown action type: {action}")
            return None

    def execute_market_order(self, symbol, order_type, volume, price, comment, sl, tp, magic):
        """
        Execute a market order.

        Parameters:
            symbol (str): The trading symbol.
            order_type (int): The order type constant.
            volume (float): The trade volume.
            price (float): The order price.
            comment (str): Order comment.
            sl (float): Stop Loss price.
            tp (float): Take Profit price.
            magic (int): Magic number.

        Returns:
            dict: Order execution result.
        """
        direction = self.get_trade_direction(order_type)
        if not direction:
            self.set_last_error(RES_E_INVALID_PARAMS, f"Invalid order type: {order_type}")
            return None

        current_prices = self.get_current_price(symbol)
        if not current_prices:
            self.set_last_error(RES_E_NOT_FOUND, f"No price data available for {symbol}.")
            return None

        exec_price = current_prices['ask'] if direction == 'BUY' else current_prices['bid']

        # Update account balance (simplified, ignoring leverage and margin)
        cost = exec_price * volume
        if direction == 'BUY':
            self.account['balance'] -= cost
        elif direction == 'SELL':
            self.account['balance'] += cost

        # Update equity
        self.account['equity'] = self.account['balance'] + self.account['profit']
        self.account['free_margin'] = self.account['equity'] - self.account['margin']

        # Create position
        position = {
            'ticket': self.next_ticket,
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'price': exec_price,
            'sl': sl,
            'tp': tp,
            'time': self.current_time,
            'comment': comment,
            'magic': magic,
            'profit': 0.0
        }
        self.open_positions[self.next_ticket] = position
        self.next_ticket += 1

        # Log the trade
        trade_log = copy.deepcopy(position)
        trade_log['action'] = 'OPEN'
        self.trade_logs.append(trade_log)

        # Return success
        return {
            'retcode': TRADE_RETCODE_DONE,
            'order': position['ticket']
        }

    def execute_pending_order(self, symbol, order_type, volume, price, comment, sl, tp, magic):
        """
        Execute a pending order by adding it to the pending_orders list.

        Parameters:
            symbol (str): The trading symbol.
            order_type (int): The order type constant.
            volume (float): The trade volume.
            price (float): The order price.
            comment (str): Order comment.
            sl (float): Stop Loss price.
            tp (float): Take Profit price.
            magic (int): Magic number.

        Returns:
            dict: Pending order placement result.
        """
        # Add pending order to the list
        pending_order = {
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'price': price,
            'comment': comment,
            'sl': sl,
            'tp': tp,
            'magic': magic
        }
        self.pending_orders.append(pending_order)
        # Assign a temporary ticket (could be enhanced)
        temp_ticket = self.next_ticket
        self.next_ticket += 1

        # Log the pending order
        pending_order_log = copy.deepcopy(pending_order)
        pending_order_log['ticket'] = temp_ticket
        pending_order_log['time'] = self.current_time
        pending_order_log['action'] = 'PENDING'
        self.trade_logs.append(pending_order_log)

        return {
            'retcode': TRADE_RETCODE_PLACED,
            'order': temp_ticket
        }

    def modify_order(self, request):
        """
        Modify an existing order (e.g., change SL/TP).

        Parameters:
            request (dict): Modification request containing 'position', 'sl', 'tp'.

        Returns:
            dict or None: Modification result or None if error.
        """
        position_ticket = request.get('position')
        if position_ticket not in self.open_positions:
            self.set_last_error(RES_E_NOT_FOUND, f"Position ticket {position_ticket} not found.")
            return None

        position = self.open_positions[position_ticket]
        if 'sl' in request and request['sl'] > 0.0:
            position['sl'] = request['sl']
        if 'tp' in request and request['tp'] > 0.0:
            position['tp'] = request['tp']

        self.open_positions[position_ticket] = position

        # Log the modification
        modification_log = {
            'ticket': position_ticket,
            'symbol': position['symbol'],
            'type': position['type'],
            'volume': position['volume'],
            'price': position['price'],
            'sl': position['sl'],
            'tp': position['tp'],
            'time': self.current_time,
            'comment': position['comment'],
            'magic': position['magic'],
            'action': 'MODIFY'
        }
        self.trade_logs.append(modification_log)

        return {
            'retcode': TRADE_RETCODE_DONE
        }

    def remove_pending_order(self, request):
        """
        Remove a pending order.

        Parameters:
            request (dict): Removal request containing 'order' (temporary ticket).

        Returns:
            dict or None: Removal result or None if error.
        """
        order_ticket = request.get('order')
        # Find and remove the pending order with the ticket
        for order in self.pending_orders:
            if order.get('ticket') == order_ticket:
                self.pending_orders.remove(order)

                # Log the removal
                removal_log = copy.deepcopy(order)
                removal_log['ticket'] = order_ticket
                removal_log['time'] = self.current_time
                removal_log['action'] = 'REMOVE'
                self.trade_logs.append(removal_log)

                return {
                    'retcode': TRADE_RETCODE_DONE
                }
        self.set_last_error(RES_E_NOT_FOUND, f"Pending order with ticket {order_ticket} not found.")
        return None

    def close_by_order(self, request):
        """
        Close a position by an opposite one.

        Parameters:
            request (dict): Close by order containing 'position'.

        Returns:
            dict or None: Close by result or None if error.
        """
        position_ticket = request.get('position')
        if position_ticket not in self.open_positions:
            self.set_last_error(RES_E_NOT_FOUND, f"Position ticket {position_ticket} not found.")
            return None

        # Implement close by logic here (simplified)
        # For this example, we'll just close the position
        self.close_position(position_ticket)

        return {
            'retcode': TRADE_RETCODE_DONE
        }

    def get_current_price(self, symbol):
        """
        Get the current price for a symbol based on current_time and advance_timeframe.

        Parameters:
            symbol (str): The trading symbol.

        Returns:
            dict or None: Current bid and ask prices or None if error.
        """
        tf_name = self.advance_timeframe  # Use the advance timeframe
        if symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        current_bar = df[df['time'] <= self.current_time].tail(1)
        if current_bar.empty:
            # Use the last known price if available
            current_bar = df.tail(1)
            if current_bar.empty:
                self.set_last_error(RES_E_NOT_FOUND, f"No price data available for {symbol}.")
                return None

        bid = current_bar['close'].iloc[0]  # Simplified bid
        ask = bid + (current_bar['spread'].iloc[0] * 0.0001)  # Simplified ask based on spread
        return {'bid': bid, 'ask': ask}

    def get_trade_direction(self, order_type):
        """
        Get trade direction based on order type.

        Parameters:
            order_type (int): The order type constant.

        Returns:
            str or None: 'BUY' or 'SELL' or None if invalid.
        """
        if order_type in [ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_BUY_STOP, ORDER_TYPE_BUY_STOP_LIMIT]:
            return 'BUY'
        elif order_type in [ORDER_TYPE_SELL, ORDER_TYPE_SELL_LIMIT, ORDER_TYPE_SELL_STOP, ORDER_TYPE_SELL_STOP_LIMIT]:
            return 'SELL'
        else:
            return None

    def advance_time_step(self):
        """
        Advance the simulation time by one minute.
        """
        # Determine the next timestamp across all symbols and timeframes
        next_times = []
        for symbol, tfs in self.symbols_data.items():
            for tf, df in tfs.items():
                future_bars = df[df['time'] > self.current_time]
                if not future_bars.empty:
                    next_times.append(future_bars['time'].iloc[0])

        if not next_times:
            # No more data to process
            self.current_time = self.end_time
            return False

        next_time = min(next_times)
        if next_time <= self.current_time:
            # Prevent infinite loop
            self.set_last_error(RES_E_INTERNAL_FAIL, "Next time step is not ahead of current time.")
            return False

        # Advance time
        self.current_time = next_time

        # Execute trade logic (process pending orders, update account)
        self.execute_trade_logic()

        # Log account status if a new hour has started
        self.log_account_status()

        return True

    def execute_trade_logic(self):
        """
        Process pending orders and update account metrics.
        """
        self.process_pending_orders()
        self.update_account_metrics()

    def process_pending_orders(self):
        """
        Process pending orders based on current price.
        """
        to_remove = []
        for order in self.pending_orders:
            symbol = order.get('symbol')
            order_type = order.get('type')
            price = order.get('price')
            volume = order.get('volume', 0.01)
            comment = order.get('comment', '')
            sl = order.get('sl', 0.0)
            tp = order.get('tp', 0.0)
            magic = order.get('magic', 0)

            current_prices = self.get_current_price(symbol)
            if not current_prices:
                continue  # Skip if no price data

            bid = current_prices['bid']
            ask = current_prices['ask']

            # Determine if pending order conditions are met
            if order_type == ORDER_TYPE_BUY_LIMIT:
                if ask <= price:
                    # Execute BUY market order
                    result = self.execute_market_order(symbol, ORDER_TYPE_BUY, volume, price, comment, sl, tp, magic)
                    if result and result['retcode'] == TRADE_RETCODE_DONE:
                        to_remove.append(order)
            elif order_type == ORDER_TYPE_SELL_LIMIT:
                if bid >= price:
                    # Execute SELL market order
                    result = self.execute_market_order(symbol, ORDER_TYPE_SELL, volume, price, comment, sl, tp, magic)
                    if result and result['retcode'] == TRADE_RETCODE_DONE:
                        to_remove.append(order)
            elif order_type == ORDER_TYPE_BUY_STOP:
                if ask >= price:
                    # Execute BUY market order
                    result = self.execute_market_order(symbol, ORDER_TYPE_BUY, volume, price, comment, sl, tp, magic)
                    if result and result['retcode'] == TRADE_RETCODE_DONE:
                        to_remove.append(order)
            elif order_type == ORDER_TYPE_SELL_STOP:
                if bid <= price:
                    # Execute SELL market order
                    result = self.execute_market_order(symbol, ORDER_TYPE_SELL, volume, price, comment, sl, tp, magic)
                    if result and result['retcode'] == TRADE_RETCODE_DONE:
                        to_remove.append(order)

        # Remove executed pending orders
        for order in to_remove:
            self.pending_orders.remove(order)

    def update_account_metrics(self):
        """
        Update account metrics based on open positions.
        """
        total_profit = 0.0
        for pos in self.open_positions.values():
            current_prices = self.get_current_price(pos['symbol'])
            if not current_prices:
                continue  # Skip if no price data

            bid = current_prices['bid']
            ask = current_prices['ask']

            if pos['type'] in [ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_BUY_STOP, ORDER_TYPE_BUY_STOP_LIMIT]:
                profit = (bid - pos['price']) * pos['volume']
            elif pos['type'] in [ORDER_TYPE_SELL, ORDER_TYPE_SELL_LIMIT, ORDER_TYPE_SELL_STOP, ORDER_TYPE_SELL_STOP_LIMIT]:
                profit = (pos['price'] - ask) * pos['volume']
            else:
                profit = 0.0

            pos['profit'] = profit
            total_profit += profit

        self.account['profit'] = total_profit
        self.account['equity'] = self.account['balance'] + self.account['profit']
        self.account['free_margin'] = self.account['equity'] - self.account['margin']

    def close_all_positions(self):
        """
        Close all open positions.
        """
        for ticket in list(self.open_positions.keys()):
            self.close_position(ticket)

    def close_position(self, ticket):
        """
        Close a single position.

        Parameters:
            ticket (int): The ticket number of the position to close.
        """
        if ticket not in self.open_positions:
            self.set_last_error(RES_E_NOT_FOUND, f"Position ticket {ticket} not found.")
            return

        position = self.open_positions[ticket]
        direction = 'SELL' if position['type'] in [
            ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_BUY_STOP, ORDER_TYPE_BUY_STOP_LIMIT
        ] else 'BUY'

        current_prices = self.get_current_price(position['symbol'])
        if not current_prices:
            self.set_last_error(RES_E_NOT_FOUND, f"No price data available for {position['symbol']}.")
            return

        exec_price = current_prices['bid'] if direction == 'SELL' else current_prices['ask']

        # Update account balance (simplified)
        cost = exec_price * position['volume']
        if direction == 'BUY':
            self.account['balance'] -= cost
        elif direction == 'SELL':
            self.account['balance'] += cost

        # Calculate profit
        if direction == 'BUY':
            profit = (exec_price - position['price']) * position['volume']
        else:
            profit = (position['price'] - exec_price) * position['volume']

        self.account['profit'] += profit
        self.account['equity'] = self.account['balance'] + self.account['profit']
        self.account['free_margin'] = self.account['equity'] - self.account['margin']

        # Move position to closed_positions
        closed_position = copy.deepcopy(position)
        closed_position['close_datetime'] = self.current_time
        closed_position['profit'] = profit
        self.closed_positions.append(closed_position)

        # Log the trade closure
        trade_log = {
            'ticket': ticket,
            'symbol': position['symbol'],
            'type': position['type'],
            'volume': position['volume'],
            'price': exec_price,
            'sl': position['sl'],
            'tp': position['tp'],
            'time': self.current_time,
            'comment': position['comment'],
            'magic': position['magic'],
            'profit': profit,
            'action': 'CLOSE'
        }
        self.trade_logs.append(trade_log)

        # Remove from open_positions
        del self.open_positions[ticket]

    def get_next_timestamp(self):
        """
        Get the next timestamp across all symbols and timeframes.

        Returns:
            datetime: The next timestamp or current_time if no future data.
        """
        next_times = []
        for symbol, tfs in self.symbols_data.items():
            for tf, df in tfs.items():
                future_bars = df[df['time'] > self.current_time]
                if not future_bars.empty:
                    next_times.append(future_bars['time'].iloc[0])

        return min(next_times) if next_times else self.current_time

    def step_simulation(self):
        """
        Advance the simulation by fixed time step.

        Returns:
            bool: True if simulation continues, False otherwise.
        """
        # Advance time by fixed time_step
        self.current_time += self.time_step

        if self.current_time > self.end_time:
            return False  # End of backtest

        # Process pending orders
        self.execute_trade_logic()

        # Log account status if a new hour has started
        self.log_account_status()

        return True

    def log_account_status(self):
        """
        Log the account status at each simulated hour.
        """
        current_hour = self.current_time.hour
        if current_hour != self.previous_hour:
            # Calculate margin level
            margin_level = (self.account['equity'] / self.account['margin']) * 100 if self.account['margin'] > 0 else 0.0

            account_doc = {
                'datetime': self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
                'open_trades': len(self.open_positions),
                'balance': self.account['balance'],
                'equity': self.account['equity'],
                'margin': self.account['margin'],
                'margin_level': margin_level
            }
            self.account_docs.append(account_doc)

            # Update previous_hour
            self.previous_hour = current_hour

    def export_logs(self):
        """
        Export trade logs and account documentation to CSV files.
        """
        # Export trade logs
        trades_df = pd.DataFrame(self.trade_logs)
        trades_csv_path = os.path.join(self.run_folder, "trade_logs.csv")
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"Trade logs exported to {trades_csv_path}")

        # Export account documentation
        account_df = pd.DataFrame(self.account_docs)
        account_csv_path = os.path.join(self.run_folder, "account_documentation.csv")
        account_df.to_csv(account_csv_path, index=False)
        print(f"Account documentation exported to {account_csv_path}")

        # Generate and save graph
        self.generate_account_graph()

    def generate_account_graph(self):
        """
        Generate and save a graph of account balance and equity over time using Plotly.
        """
        if not self.account_docs:
            print("No account documentation available to generate graph.")
            return

        account_df = pd.DataFrame(self.account_docs)
        account_df['datetime'] = pd.to_datetime(account_df['datetime'])
        account_df.sort_values('datetime', inplace=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=account_df['datetime'],
            y=account_df['balance'],
            mode='lines',
            name='Balance'
        ))

        fig.add_trace(go.Scatter(
            x=account_df['datetime'],
            y=account_df['equity'],
            mode='lines',
            name='Equity'
        ))

        fig.update_layout(
            title='Account Balance and Equity Over Time',
            xaxis_title='Time',
            yaxis_title='Amount ($)',
            xaxis=dict(rangeslider_visible=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )

        # Save the figure as an HTML file
        graph_path_html = os.path.join(self.run_folder, "account_balance_equity.html")
        fig.write_html(graph_path_html)
        print(f"Interactive account balance and equity graph saved to {graph_path_html}")

        # Optionally, save as a static image (requires kaleido)
        graph_path_png = os.path.join(self.run_folder, "account_balance_equity.png")
        try:
            fig.write_image(graph_path_png)
            print(f"Account balance and equity graph saved to {graph_path_png}")
        except Exception as e:
            print(f"Could not save graph as image. Install 'kaleido' to enable image export. Error: {e}")
            

    def run_backtest(self):
        """
        Run the backtest until the end time is reached.
        """
        print(f"Starting backtest from {self.current_time} to {self.end_time}")
        while self.current_time < self.end_time:
            proceed = self.step_simulation()
            if not proceed:
                print(f"Backtest stopped due to error: {self.last_error_description}")
                break
        print("Backtest completed.")
        print(f"Final balance: {self.account['balance']}")
        print(f"Final equity: {self.account['equity']}")
        print(f"Total profit: {self.account['profit']}")

        # Export logs
        self.export_logs()

    def last_error(self):
        """
        Simulate MT5.last_error() function.

        Returns:
            tuple: (error_code, error_description)
        """
        return (self.last_error_code, self.last_error_description)

    def set_last_error(self, code, description):
        """
        Set the last error.

        Parameters:
            code (int): Error code.
            description (str): Error description.
        """
        self.last_error_code = code
        self.last_error_description = description

    def shutdown(self):
        """
        Simulate MT5.shutdown() function.
        """
        # For backtesting, shutdown might reset the environment
        self.current_time = None
        self.open_positions.clear()
        self.closed_positions.clear()
        self.pending_orders.clear()
        self.account = {
            'balance': 100000.0,
            'equity': 100000.0,
            'margin': 0.0,
            'free_margin': 100000.0,
            'profit': 0.0
        }
        self.set_last_error(RES_S_OK, "Shutdown complete.")

# Instantiate a global backtest instance - why do I need this?
backtest = MT5Backtest()

# ----------------------------
# Exported Functions
# ----------------------------

def account_info():
    """
    Simulate MT5.account_info() function.

    Returns:
        dict: Account information.
    """
    return backtest.account_info()

def copy_rates(symbol, timeframe, count):
    """
    Simulate MT5.copy_rates() function.

    Parameters:
        symbol (str): The trading symbol.
        timeframe (int): The timeframe constant.
        count (int): Number of bars to copy.

    Returns:
        np.recarray or None: Array of rates up to current_time or None if error.
    """
    return backtest.copy_rates(symbol, timeframe, count)

def copy_rates_from(symbol, timeframe, datetime_from, count):
    """
    Simulate MT5.copy_rates_from() function.

    Parameters:
        symbol (str): The trading symbol.
        timeframe (int): The timeframe constant.
        datetime_from (datetime): Start datetime.
        count (int): Number of bars to copy.

    Returns:
        np.recarray or None: Array of rates or None if error.
    """
    return backtest.copy_rates_from(symbol, timeframe, datetime_from, count)

def copy_rates_from_pos(symbol, timeframe, pos, count):
    """
    Simulate MT5.copy_rates_from_pos() function.

    Parameters:
        symbol (str): The trading symbol.
        timeframe (int): The timeframe constant.
        pos (int): Starting position.
        count (int): Number of bars to copy.

    Returns:
        np.recarray or None: Array of rates or None if error.
    """
    return backtest.copy_rates_from_pos(symbol, timeframe, pos, count)

def order_send(request):
    """
    Simulate MT5.order_send() function.

    Parameters:
        request (dict): Order request parameters.

    Returns:
        dict or None: Order execution result or None if error.
    """
    return backtest.order_send(request)

def positions_get(ticket=None):
    """
    Simulate MT5.positions_get() function.

    Parameters:
        ticket (int, optional): Specific ticket number to retrieve.

    Returns:
        dict or list or None: Single position dict, list of positions, or None if not found.
    """
    return backtest.positions_get(ticket)

def symbol_info_tick(symbol):
    """
    Simulate MT5.symbol_info_tick() function.

    Parameters:
        symbol (str): The trading symbol.

    Returns:
        dict or None: Tick information or None if error.
    """
    return backtest.symbol_info_tick(symbol)

def symbol_info(symbol):
    """
    Simulate MT5.symbol_info() function.

    Parameters:
        symbol (str): The trading symbol.

    Returns:
        dict: Symbol information.
    """
    return backtest.symbol_info(symbol)

def symbol_select(symbol, select=True):
    """
    Simulate MT5.symbol_select() function.

    Parameters:
        symbol (str): The trading symbol.
        select (bool): Whether to select or deselect the symbol.

    Returns:
        bool: Success status.
    """
    return backtest.symbol_select(symbol, select)

def history_deals_get(from_date, to_date):
    """
    Simulate MT5.history_deals_get() function.

    Parameters:
        from_date (datetime): Start datetime.
        to_date (datetime): End datetime.

    Returns:
        list or None: List of deal dictionaries or None if no deals found.
    """
    return backtest.history_deals_get(from_date, to_date)

def last_error():
    """
    Simulate MT5.last_error() function.

    Returns:
        tuple: (error_code, error_description)
    """
    return backtest.last_error()

def shutdown():
    """
    Simulate MT5.shutdown() function.
    """
    backtest.shutdown()

def run_backtest():
    """
    Function to run the backtest externally.
    """
    backtest.run_backtest()