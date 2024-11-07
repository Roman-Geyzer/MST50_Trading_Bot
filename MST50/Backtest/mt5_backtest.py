# mt5_backtest.py
"""
This module contains the MT5Backtest class that simulates the MetaTrader 5 client for backtesting purposes.
The class provides methods to simulate the MetaTrader 5 client functions for backtesting trading strategies.
The class is designed to be used with the MT5Strategy class to backtest trading strategies.
functions:
    MT5Backtest: A class to simulate
    the MetaTrader 5 (MT5) client for backtesting purposes.
    extract_backtest_parameters: Extract symbols, timeframes, start_time, end_time, and time_step from strategies.
    load_data: Load historical data from CSV files into symbols_data.
    get_timeframe_name: Get the timeframe name from its value.
    copy_rates: Simulate MT5.copy_rates() function.
    copy_rates_from_pos: Simulate MT5.copy_rates_from_pos() function.
    copy_rates_from: Simulate MT5.copy_rates_from() function.
    account_info: Simulate MT5.account_info() function.
    positions_get: Simulate MT5.positions_get() function.
    symbol_info_tick: Simulate MT5.symbol_info_tick() function.
    symbol_info: Simulate MT5.symbol_info() function.
    symbol_select: Simulate MT5.symbol_select() function.
    history_deals_get: Simulate MT5.history_deals_get() function.
    order_send: Simulate MT5.order_send() function.
    execute_market_order: Execute a market order.
    execute_pending_order: Execute a pending order by adding it to the pending_orders list.
    modify_order: Modify an existing order (e.g., change SL/TP).
    remove_pending_order: Remove a pending order.
    close_by_order: Close a position by an opposite one.
    get_current_price: Get the current price for a symbol based on current_time and advance_timeframe.
    get_trade_direction: Get trade direction based on order type.
    advance_time_step: Advance the simulation time by one minute.
    execute_trade_logic: Process pending orders and update account metrics.
    process_pending_orders: Process pending orders based on current price.
    update_account_metrics: Update account metrics based on open positions.
    close_all_positions: Close all open positions.
    close_position: Close a single position.
    simulate_target_hit_close: Check if any open positions have hit their SL or TP and close them accordingly.
    get_current_bar_data: Get the current bar data for a symbol based on current_time and advance_timeframe.
    initialize_current_tick_indices: Initialize current_tick_index for each symbol and timeframe based on self.start_time.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

import plotly.graph_objects as go
from .mt5_backtest_constants import *
from .time_backtest import TimeBar, TradeHour



# days set to end backtest - x days ago from today
backtest_end_relative_to_today = 180


constants = get_constants()
# Expose constants
TIMEFRAMES = constants['TIMEFRAMES']
ORDER_TYPES = constants['ORDER_TYPES']
TRADE_ACTIONS = constants['TRADE_ACTIONS']
ORDER_TIME = constants['ORDER_TIME']
ORDER_FILLING = constants['ORDER_FILLING']
TRADE_RETCODES = constants['TRADE_RETCODES']


# ----------------------------
# MT5Backtest Class Definition
# ----------------------------

backtest = None  # Global backtest instance to be initialized later

drive = "x:" if os.name == 'nt' else "/Volumes/TM"

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
        if strategies: # since there is one gloabl instance of this class with no strategies - this will not be executed
            self.extract_backtest_parameters()
            self.load_data()

            # Initialize simulation parameters
            self.current_time = self.start_time
            self.end_time = self.end_time
            # Initialize previous hour to track hourly logging
            self.previous_hour = self.current_time.hour
        else:
            self.advance_timeframe = None
            self.current_time = None
            self.end_time = None
            self.time_step = None
            self.previous_hour = None

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
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(self.output_dir, f"Backtest_{start_time_str}")
        os.makedirs(self.run_folder, exist_ok=True)

        # Initialize trade logs and account documentation
        self.trade_logs = []      # List to store trade dictionaries
        self.account_docs = []    # List to store account documentation dictionaries

        # Initialize current tick indices for each symbol and timeframe
        self.current_tick_index = {}
        for symbol, tfs in self.symbols_data.items():
            self.current_tick_index[symbol] = {}
            for tf_name in tfs.keys():
                self.current_tick_index[symbol][tf_name] = 0  # Start at the first bar

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
            # Add higher and lower timeframes from strategy, ignoring None
            higher_tf_str = get_mt5_tf_str(strategy.higher_timeframe)
            lower_tf_str = get_mt5_tf_str(strategy.lower_timeframe)
            if higher_tf_str is not None:
                timeframes_set.add(higher_tf_str)
            if lower_tf_str is not None:
                timeframes_set.add(lower_tf_str)
            # Collect backtest start date and time step
            backtest_start_dates.append(strategy.backtest_start_date)
            backtest_time_steps.append(strategy.backtest_tf)

        self.symbols = list(symbols_set)
        self.timeframes = list(timeframes_set)

        # Use the earliest start date among strategies
        self.start_time = min(backtest_start_dates)

        # Set end date as X days before today
        self.end_time = datetime.now() - timedelta(days=backtest_end_relative_to_today)

        # Use the backtest time step (candle advance) from the strategies
        # Assuming all strategies have the same backtest time frame
        backtest_timeframe = backtest_time_steps[0]  # Assuming all are the same

        # Convert the timeframe constant to its string representation
        backtest_timeframe_str = get_mt5_tf_str(backtest_timeframe)
        self.advance_timeframe = backtest_timeframe_str  # Store the timeframe string

        # Map the timeframe string to a timedelta
        self.time_step = timeframe_to_timedelta(backtest_timeframe_str)


    def load_data(self):
        """
        Load historical data from CSV files into symbols_data.
        Only loads data for specified symbols and timeframes, starting from self.start_time.
        """
        if not self.symbols or not self.timeframes:
            print("No symbols or timeframes specified for data loading.")
            return

        required_columns = {'time', 'open', 'high', 'low', 'close', 'spread'}

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

            df = pd.read_csv(filepath)
            
            # Check if 'time' column exists
            if 'time' not in df.columns:
                print(f"Error: 'time' column missing in {filename}. Skipping.")
                continue

            # **Remove utc=True to keep 'time' timezone-naive**
            try:
                df['time'] = pd.to_datetime(df['time'], errors='raise')  # Removed utc=True
            except Exception as e:
                print(f"Error parsing 'time' column in {filename}: {e}. Skipping.")
                continue

            # Check for required columns
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                print(f"Error: Missing columns {missing} in {filename}. Skipping.")
                continue

            # Remove rows with any NaT in 'time'
            if df['time'].isnull().any():
                print(f"Warning: Some 'time' entries could not be parsed in {filename}. They will be dropped.")
                df = df.dropna(subset=['time'])

            # Sort by 'time' ascending
            df.sort_values('time', inplace=True)

            # Filter data to include only rows where 'time' >= self.start_time
            if self.start_time:
                initial_row_count = len(df)
                df = df[df['time'] >= self.start_time]
                filtered_row_count = len(df)
                print(f"    Filtered data for {symbol} on timeframe {tf_name}: {filtered_row_count} out of {initial_row_count} bars retained (from {self.start_time}).")
            else:
                print(f"    No start_time specified. Loading all data for {symbol} on timeframe {tf_name}.")

            # Set 'time' as index for faster access
            df.set_index('time', inplace=True)  # Set 'time' as index
            df.drop_duplicates(inplace=True)    # Remove any duplicate timestamps
            # Reset index to keep 'time' as a column while also having it as the index
            df.reset_index(inplace=True)

            # **Precompute NumPy arrays for faster access**
            df['time_np'] = df['time'].astype('int64') // 10**9  # Convert to UNIX timestamp
            df['close_np'] = df['close'].values
            df['spread_np'] = df['spread'].values

            # Ensure symbols_data[symbol] is initialized as a dict
            if symbol not in self.symbols_data:
                self.symbols_data[symbol] = {}

            # Assign the DataFrame to the specific timeframe
            self.symbols_data[symbol][tf_name] = df

            print(f"    Loaded data for {symbol} on timeframe {tf_name} with {len(df)} bars.")

        # Initialize current_tick_index after loading all data
        self.initialize_current_tick_indices()
    
    def initialize_current_tick_indices(self):
        """
        Initialize the current_tick_index for each symbol and timeframe.
        """
        for symbol, tfs in self.symbols_data.items():
            for tf_name in tfs:
                self.current_tick_index[symbol][tf_name] = 0

    def initialize_current_tick_indices(self):
        """
        Initialize current_tick_index for each symbol and timeframe based on self.start_time.
        Sets the current_tick_index to the first bar at or after self.start_time.
        """
        self.current_tick_index = {}
        for symbol, tfs in self.symbols_data.items():
            self.current_tick_index[symbol] = {}
            for tf_name, df in tfs.items():
                # Find the first index where 'time' >= self.start_time
                idx = df.index[df['time'] >= self.start_time].tolist()
                if idx:
                    first_valid_index = idx[0]
                    self.current_tick_index[symbol][tf_name] = first_valid_index
                    print(f"    Initialized current_tick_index for {symbol} on {tf_name}: {first_valid_index}")
                else:
                    # If no data after self.start_time, set index to len(df) to indicate no data
                    self.current_tick_index[symbol][tf_name] = len(df)
                    print(f"    No data after start_time for {symbol} on {tf_name}. Setting current_tick_index to {len(df)}")    

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
        current_index = self.current_tick_index[symbol][tf_name]

        if current_index < 0 or current_index >= len(df):
            self.set_last_error(RES_E_NOT_FOUND, f"Current index {current_index} out of range for {symbol}, {tf_name}")
            return None

        # Determine the start and end indices for slicing
        start_index = max(current_index - count + 1, 0)
        end_index = current_index + 1  # +1 because slicing is exclusive at the end

        rates = df.iloc[start_index:end_index]
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

        # Use iloc to slice based on position
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
        return self.account

    def positions_get(self, ticket=None):
        """
        Simulate MT5.positions_get() function.

        Parameters:
            ticket (int, optional): Specific ticket number to retrieve.

        Returns:
            dict or list or None: Single position dict, list of positions, or None if not found.
        """
        if ticket is not None:
            return self.open_positions.get(ticket, None)
        else:
            return (self.open_positions.values()) if self.open_positions else None

    def symbol_info_tick(self, symbol):
        """
        Simulate MT5.symbol_info_tick() function.

        Parameters:
            symbol (str): The trading symbol.

        Returns:
            dict or None: Tick information or None if error.
        """
        tf_name = self.advance_timeframe  # Use the advance timeframe

        # Quick existence check using 'in' for both symbol and timeframe
        if symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        current_index = self.current_tick_index[symbol][tf_name]

        # Check if current_index is within bounds
        if current_index >= len(df):
            self.set_last_error(RES_E_NOT_FOUND, f"No more tick data available for {symbol}.")
            return None

        # **Access precomputed NumPy arrays for faster retrieval**
        tick_time = df['time_np'].iloc[current_index]
        bid = df['close_np'].iloc[current_index]
        spread = df['spread_np'].iloc[current_index]
        ask = bid + (spread * 0.0001)
        last = bid

        tick = {
            'time': int(tick_time),  # UNIX timestamp as integer
            'bid': bid,               # Close price as bid
            'ask': ask,               # Close price plus scaled spread as ask
            'last': last              # Close price as last
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
        if 'JPY' in symbol:
            digits = 3
        else:
            digits = 5

        # Simplified symbol information
        info = {
            'name': symbol,
            'path': symbol,
            'description': symbol,
            'digits': digits,
            'point': 10 ** -digits,
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
                deals.append(pos)
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
        elif action == TRADE_ACTION_SLTP:
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
        trade_log = position
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
        pending_order_log = pending_order
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
                removal_log = order
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
        Advance the simulation time to the next data point.
        Returns:
            bool: True if simulation continues, False otherwise.
        """
        # Determine the next timestamp across all symbols and timeframes
        next_times = []
        for symbol, tfs in self.symbols_data.items():
            for tf_name, df in tfs.items():
                current_index = self.current_tick_index[symbol][tf_name]
                if current_index + 1 < len(df):
                    next_bar_time = df.iloc[current_index + 1]['time']
                    if isinstance(next_bar_time, (int, float)):
                        # Convert UNIX timestamp to datetime if necessary
                        next_bar_time = datetime.fromtimestamp(next_bar_time)
                    if next_bar_time > self.current_time:
                        next_times.append(next_bar_time)

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

        return True

    def execute_trade_logic(self):
        """
        Process pending orders and update account metrics.
        """
        # Simulate SL/TP hits before processing pending orders
        self.simulate_target_hit_close()

        self.process_pending_orders()
        self.update_account_metrics()

    def simulate_target_hit_close(self):
        """
        Check if any open positions have hit their SL or TP and close them accordingly.
        """
        tickets_to_check = list(self.open_positions.keys())  # Create a list to avoid runtime errors
        for ticket in tickets_to_check:
            position = self.open_positions[ticket]
            symbol = position['symbol']
            order_type = position['type']
            sl = position['sl']
            tp = position['tp']

            # Get current bar data for the symbol
            bar_data = self.get_current_bar_data(symbol)
            if bar_data is None:
                continue  # Skip if no bar data available

            high = bar_data['high']
            low = bar_data['low']

            # Determine if SL or TP was hit
            sl_hit = False
            tp_hit = False

            if order_type == ORDER_TYPE_BUY:
                if sl and low <= sl:
                    sl_hit = True
                    exit_price = sl  # Exit at SL price
                elif tp and high >= tp:
                    tp_hit = True
                    exit_price = tp  # Exit at TP price
            elif order_type == ORDER_TYPE_SELL:
                if sl and high >= sl:
                    sl_hit = True
                    exit_price = sl  # Exit at SL price
                elif tp and low <= tp:
                    tp_hit = True
                    exit_price = tp  # Exit at TP price

            if sl_hit or tp_hit:
                # Close the position
                self.close_position(ticket, exit_price, reason='SL' if sl_hit else 'TP')

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
        if not self.open_positions:
            self.account['profit'] = 0.0
        else:
            # Create a DataFrame from open positions
            positions_df = pd.DataFrame(self.open_positions.values())
            # Get current prices for all symbols
            symbols = positions_df['symbol'].unique()
            current_prices = {symbol: self.get_current_price(symbol) for symbol in symbols}

            # Calculate profit for each position
            def calculate_profit(row):
                price_data = current_prices.get(row['symbol'], {})
                bid = price_data.get('bid', row['price'])
                ask = price_data.get('ask', row['price'])
                if row['type'] in [ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_BUY_STOP, ORDER_TYPE_BUY_STOP_LIMIT]:
                    return (bid - row['price']) * row['volume']
                elif row['type'] in [ORDER_TYPE_SELL, ORDER_TYPE_SELL_LIMIT, ORDER_TYPE_SELL_STOP, ORDER_TYPE_SELL_STOP_LIMIT]:
                    return (row['price'] - ask) * row['volume']
                else:
                    return 0.0

            positions_df['profit'] = positions_df.apply(calculate_profit, axis=1)
            total_profit = positions_df['profit'].sum()

            # Update account
            self.account['profit'] = total_profit
            self.account['equity'] = self.account['balance'] + self.account['profit']
            self.account['free_margin'] = self.account['equity'] - self.account['margin']

    def close_all_positions(self):
        """
        Close all open positions.
        """
        for ticket in list(self.open_positions.keys()):
            self.close_position(ticket)
    
    def get_current_bar_data(self, symbol):
        """
        Get the current bar data for the symbol at current_time.

        Parameters:
            symbol (str): The trading symbol.

        Returns:
            dict or None: Bar data or None if not found.
        """
        tf_name = self.advance_timeframe  # Use the advance timeframe
        if symbol not in self.symbols_data or tf_name not in self.symbols_data[symbol]:
            self.set_last_error(RES_E_NOT_FOUND, f"Symbol or timeframe not found: {symbol}, {tf_name}")
            return None

        df = self.symbols_data[symbol][tf_name]
        current_bar = df.iloc[self.current_tick_index[symbol][tf_name]]
        if current_bar is None:
            return None

        bar = current_bar.to_dict()
        return bar

    def close_position(self, ticket, exit_price=None, reason='CLOSE'):
        """
        Close a single position.

        Parameters:
            ticket (int): The ticket number of the position to close.
            exit_price (float): The price at which the position is closed.
            reason (str): The reason for closing ('CLOSE', 'SL', 'TP').
        """
        if ticket not in self.open_positions:
            self.set_last_error(RES_E_NOT_FOUND, f"Position ticket {ticket} not found.")
            return

        position = self.open_positions[ticket]
        order_type = position['type']
        volume = position['volume']
        entry_price = position['price']
        symbol = position['symbol']

        # Get current prices if exit_price not provided
        if exit_price is None:
            current_prices = self.get_current_price(symbol)
            if not current_prices:
                self.set_last_error(RES_E_NOT_FOUND, f"No price data available for {symbol}.")
                return
            if order_type == ORDER_TYPE_BUY:
                exit_price = current_prices['bid']
            elif order_type == ORDER_TYPE_SELL:
                exit_price = current_prices['ask']

        # Calculate profit
        if order_type == ORDER_TYPE_BUY:
            profit = (exit_price - entry_price) * volume
        elif order_type == ORDER_TYPE_SELL:
            profit = (entry_price - exit_price) * volume
        else:
            profit = 0.0

        # Update account balance
        self.account['balance'] += profit
        self.account['profit'] -= position['profit']  # Remove unrealized profit
        self.account['equity'] = self.account['balance'] + self.account['profit']
        self.account['free_margin'] = self.account['equity'] - self.account['margin']

        # Move position to closed_positions
        closed_position = position
        closed_position['close_datetime'] = self.current_time
        closed_position['close_price'] = exit_price
        closed_position['profit'] = profit
        self.closed_positions.append(closed_position)

        # Log the trade closure
        trade_log = {
            'ticket': ticket,
            'symbol': symbol,
            'type': order_type,
            'volume': volume,
            'price': exit_price,
            'sl': position['sl'],
            'tp': position['tp'],
            'time': self.current_time,
            'comment': position['comment'],
            'magic': position['magic'],
            'profit': profit,
            'action': reason
        }
        self.trade_logs.append(trade_log)

        # Remove from open_positions
        del self.open_positions[ticket]

    def step_simulation(self):
        """
        Advance the simulation time to the next data point.
        Returns:
            bool: True if simulation continues, False otherwise.
        """
        proceed = self.advance_time_step()
        if not proceed:
            return False  # End of backtest or error

        # Update current_tick_index for each symbol and timeframe
        for symbol, tfs in self.symbols_data.items():
            for tf_name, df in tfs.items():
                current_index = self.current_tick_index[symbol][tf_name]
                if current_index + 1 < len(df):
                    next_bar_time = df.iloc[current_index + 1]['time']
                    if next_bar_time <= self.current_time:
                        self.current_tick_index[symbol][tf_name] += 1

        # Execute trade logic (process pending orders, update account)
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

        # Optionally, display the graph
        fig.show()
            

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



# ----------------------------
# Exported Functions
# ----------------------------

# Global backtest instance - will support the exported functions
def initialize_backtest(strategies):
    """
    Initialize the global backtest instance with the provided strategies.

    Parameters:
        strategies (dict): Dictionary of strategy instances.
    """
    global backtest
    backtest = MT5Backtest(strategies=strategies)
    print(f"Backtest params: {backtest.__dict__}")


def account_info():
    """
    Simulate MT5.account_info() function.

    Returns:
        dict: Account information.
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
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
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
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
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
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
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    return backtest.copy_rates_from_pos(symbol, timeframe, pos, count)

def order_send(request):
    """
    Simulate MT5.order_send() function.

    Parameters:
        request (dict): Order request parameters.

    Returns:
        dict or None: Order execution result or None if error.
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    return backtest.order_send(request)

def positions_get(ticket=None):
    """
    Simulate MT5.positions_get() function.

    Parameters:
        ticket (int, optional): Specific ticket number to retrieve.

    Returns:
        dict or list or None: Single position dict, list of positions, or None if not found.
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    return backtest.positions_get(ticket)

def symbol_info_tick(symbol):
    """
    Simulate MT5.symbol_info_tick() function.

    Parameters:
        symbol (str): The trading symbol.

    Returns:
        dict or None: Tick information or None if error.
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
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
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
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
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    return backtest.history_deals_get(from_date, to_date)

def last_error():
    """
    Simulate MT5.last_error() function.

    Returns:
        tuple: (error_code, error_description)
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    return backtest.last_error()

def shutdown():
    """
    Simulate MT5.shutdown() function.
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    backtest.shutdown()
    quit()

def run_backtest(strategies, symbols):
    """
    Function to run the backtest externally.
    Parameters:
        strategies (dict): Dictionary of strategy instances.
        symbols (dict): Dictionary of symbol instances.
    """
    if backtest is None:
        raise RuntimeError("Backtest instance is not initialized. Call initialize_backtest() first.")
    
    from ..run_bot import on_minute

    # Initialize TradeHour and TimeBar with backtest
    trade_hour = TradeHour(backtest)
    time_bar = TimeBar(backtest)

    print(f"Starting backtest from {backtest.current_time} to {backtest.end_time}")
    while backtest.current_time < backtest.end_time:
        # Advance time by time_step
        proceed = backtest.step_simulation()
        if not proceed:
            print(f"Backtest stopped due to error: {backtest.last_error_description}")
            break

        # Call on_minute with current simulation time
        on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict=None)

    print("Backtest completed.")
    print(f"Final balance: {backtest.account['balance']}")
    print(f"Final equity: {backtest.account['equity']}")
    print(f"Total profit: {backtest.account['profit']}")

    # Export logs
    backtest.export_logs()