# mt5_backtest.py

"""
This module contains the MT5Backtest class that simulates the MetaTrader 5 client for backtesting purposes.
The class provides methods to simulate the MetaTrader 5 client functions for backtesting trading strategies.
The class is designed to be used with the MT5Strategy class to backtest trading strategies.



Functions:

	MT5Backtest: A class to simulate the MetaTrader 5 (MT5) client for backtesting purposes.
	extract_backtest_parameters: Extract symbols, timeframes, start_time, data_start_time, end_time, and time_step from strategies.
	load_data: Load historical data from Parquet files into symbols_data as NumPy arrays.
	load_timeseries_data: Load timeseries data for the backtest timeframe.
	get_timeframe_name: Get the timeframe name from its value.
	copy_rates_from_pos: Optimized function to simulate MT5.copy_rates_from_pos().
	account_info: Simulate MT5.account_info() function.
	positions_get: Simulate MT5.positions_get() function.
	symbol_info_tick: Simulate MT5.symbol_info_tick() function.
	symbol_info: Simulate MT5.symbol_info() function with caching.
	symbol_select: Simulate MT5.symbol_select() function.
	history_deals_get: Simulate MT5.history_deals_get() function.
	order_send: Simulate MT5.order_send() function.
	execute_market_order: Execute a market order.
	modify_order: Modify an existing order (e.g., change SL/TP).
	close_by_order: Close a position by an opposite one.
	get_current_price: Optimized function to get the current price for a symbol.
	advance_time_step: Advance the simulation time by one timestep.
	simulate_target_hit_close: Check if SL or TP were hit and update account metrics.
	update_account_metrics: Update account metrics based on open positions.
	close_all_positions: Close all open positions.
	close_position: Close a single position.
	simulate_target_hit_close: Optimized function to check if any open positions have hit their SL or TP.
	get_current_bar_data: Get the current bar data for a symbol.
	initialize_current_tick_indices: Initialize current_tick_index for each symbol and timeframe.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import plotly.graph_objects as go
from .mt5_backtest_constants import *
from .time_backtest import TimeBar, TradeHour

import time

# Constants
backtest_end_relative_to_today = 180
leverage = 100  # Assuming leverage of 1:100
contract_size = 100_000  # For Forex standard lots
leverage_contract_ratio = contract_size/leverage

constants = get_constants()
# Expose constants
TIMEFRAMES = constants['TIMEFRAMES']
ORDER_TYPES = constants['ORDER_TYPES']
TRADE_ACTIONS = constants['TRADE_ACTIONS']
ORDER_TIME = constants['ORDER_TIME']
ORDER_FILLING = constants['ORDER_FILLING']
TRADE_RETCODES = constants['TRADE_RETCODES']

drive = "x:" if os.name == 'nt' else "/Volumes/TM"

initial_balance = 1_000_000.0

def collect_usd_currency_pairs_and_non_usd_bases(symbols):
	"""
	Collect currency pairs involving USD and ensure non-USD base currencies are paired with USD.

	Parameters:
		symbols (list): A list of symbols.

	Returns:
		set: A set of currency pairs involving USD (direct or derived).
	"""
	currency_set = set()  # Use a set to store unique currency pairs
	major_pairs = {"EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"}

	for symbol in symbols:
		quote_currency = symbol[-3:]  # Last three characters

		# If the quote currency is not USD, ensure USD quote_currency is added
		if quote_currency != "USD":
			pair_with_usd = f"USD{quote_currency}"
			currency_set.add(pair_with_usd)
			pair_with_usd = f"{quote_currency}USD"
			currency_set.add(pair_with_usd)

	# Keep only valid major pairs
	filtered_currency_set = currency_set.intersection(major_pairs)
	return filtered_currency_set

class MT5Backtest:
	"""
	A class to simulate the MetaTrader 5 (MT5) client for backtesting purposes.
	"""

	def __init__(self, strategies, data_dir="historical_data", output_dir="Backtests"):
		"""
		Initialize the backtest environment.

		Parameters:
			strategies (dict): Dictionary of strategy instances.
			data_dir (str): Directory where historical data is stored.
			output_dir (str): Directory where backtest outputs will be saved.
		"""
		self._initialized = True  # To match the singleton pattern in main.py
		self.strategies = strategies  # Store the strategies
		self.data_dir = os.path.join(drive, data_dir)
		self.symbols_data = {}  # {symbol: {timeframe: {column: numpy array}}}
		self.current_tick_index = {}  # {symbol: {timeframe: current_index}}    

		# Extract symbols, timeframes, and backtest parameters from strategies
		if strategies:
			self.extract_backtest_parameters()
			self.dtypes = None  # Will hold the predefined data types for the numpy arrays
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
			'balance': initial_balance,
			'equity': initial_balance,
			'margin': 0.0,
			'free_margin': initial_balance,
			'profit': 0.0,
			'margin_level': 0.0
		}

		# Initialize positions
		self.open_positions = {}   # {ticket: position_info}
		self.closed_positions = {}  # {ticket: position_info}

		# Ticket counter
		self.next_ticket = 1  # Starting ticket number

		# Output directory setup
		self.output_dir = os.path.join(drive, output_dir)
		start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_folder = os.path.join(self.output_dir, f"Backtest_{start_time_str}")
		os.makedirs(self.run_folder, exist_ok=True)

		# Initialize trade logs and account documentation
		self.trade_logs = []      # List to store trade dictionaries
		self.account_docs = []    # List to store account documentation dictionaries

		# Load timeseries data
		self.load_timeseries_data()

	def extract_backtest_parameters(self):
		"""
		Extract symbols, timeframes, start_time, end_time, and time_step from strategies.
		"""
		symbols_set = set()
		timeframes_set = set()
		backtest_start_dates = []
		backtest_buffered_start_dates = []
		backtest_time_steps = []
		required_columns_set = set()

		for strategy in self.strategies.values():
			symbols_set.update(strategy.symbols)
			symbols_set.update(collect_usd_currency_pairs_and_non_usd_bases(strategy.symbols))

			# Add the strategy's main timeframe
			timeframes_set.add(strategy.str_timeframe)
			timeframes_set.add(get_mt5_tf_str(strategy.backtest_tf))
			# Add higher and lower timeframes from strategy, ignoring None
			higher_tf_str = get_mt5_tf_str(strategy.higher_timeframe)
			lower_tf_str = get_mt5_tf_str(strategy.lower_timeframe)
			if higher_tf_str is not None:
				timeframes_set.add(higher_tf_str)
			if lower_tf_str is not None:
				timeframes_set.add(lower_tf_str)
			# Collect backtest start date and time step
			backtest_start_dates.append(strategy.backtest_start_date)
			orig_start_date = strategy.backtest_start_date
			if strategy.str_timeframe == 'D1':
				# set the first bar to load (time) - to be history length before the start time
				backtest_buffered_start_dates.append(orig_start_date - timedelta(days=205)) # Load 205 days of data - more than enough for D1
			else:
				# set the first bar to load (time) - to be history length before the start time
				backtest_buffered_start_dates.append(orig_start_date - timedelta(days=30)) #  Load 30 days of data - more than enough for any other timeframe
			backtest_time_steps.append(strategy.backtest_tf)

			# Collect required columns from the strategy
			required_columns_set.update(strategy.required_columns)
			required_columns_set.update(['bid', 'ask'])

		self.symbols = list(symbols_set)
		self.timeframes = list(timeframes_set)
		self.required_columns = required_columns_set  # Store the required columns

		# Use the earliest start date among strategies
		self.start_time = min(backtest_start_dates)
		self.data_start_time = min(backtest_buffered_start_dates)  # Use the buffered start date for data loading

		# Set end date as X days before today
		self.end_time = datetime.now() - timedelta(days=backtest_end_relative_to_today)

		# Use the backtest time step (candle advance) from the strategies
		# Assuming all strategies have the same backtest time frame
		backtest_timeframe = min(backtest_time_steps)

		# Convert the timeframe constant to its string representation
		backtest_timeframe_str = get_mt5_tf_str(backtest_timeframe)
		self.advance_timeframe = backtest_timeframe_str  # Store the timeframe string

		# Map the timeframe string to a timedelta
		self.time_step = timeframe_to_timedelta(backtest_timeframe_str)

	def load_data(self):
		"""
		Load historical data from Parquet files into symbols_data.
		Only loads data for specified symbols and timeframes, starting from self.data_start_time.
		"""
		if not self.symbols or not self.timeframes:  
			print("No symbols or timeframes specified for data loading.")
			return

		required_columns = self.required_columns

		for filename in os.listdir(self.data_dir):
			filepath = os.path.join(self.data_dir, filename)
			try:
				symbol, tf_name = filename.replace('.parquet', '').split('_')
			except ValueError:
				print(f"Filename {filename} does not match the required format 'Symbol_Timeframe.parquet'. Skipping.")
				continue

			# Only load data for specified symbols and timeframes
			if symbol not in self.symbols or tf_name not in self.timeframes:
				continue  # Skip unnecessary data

			# Read data from Parquet file
			try:
				df = pd.read_parquet(filepath, columns=required_columns)
			except Exception as e:
				print(f"Error reading {filename}: {e}. Skipping.")
				continue

			# Ensure 'time' column is present and convert to datetime
			if 'time' not in df.columns:
				print(f"Error: 'time' column missing in {filename}. Skipping.")
				continue

			# Convert 'time' column to datetime objects
			df['time'] = pd.to_datetime(df['time'], errors='raise')

			# Remove rows with any NaT in 'time'
			if df['time'].isnull().any():
				print(f"Warning: Some 'time' entries could not be parsed in {filename}. They will be dropped.")
				df = df.dropna(subset=['time'])

			# Filter data to include only rows where 'time' >= self.data_start_time
			if self.start_time:
				df = df[df['time'] >= self.data_start_time]
			else:
				print(f"    No start_time specified. Loading all data for {symbol} on timeframe {tf_name}.")

			# Sort by 'time' ascending (oldest to newest)
			df.sort_values('time', inplace=True, ascending=True)
			df.drop_duplicates(inplace=True)
			df.reset_index(drop=True, inplace=True)

			# Convert DataFrame columns to NumPy arrays and store in a dict
			data_dict = {}
			for col in df.columns:
				if col == 'time':
					data_dict[col] = df[col].to_numpy(dtype='object')
				else:
					data_dict[col] = df[col].to_numpy()

			# Create structured array and store it
			arrays = [data_dict[field] for field in data_dict]
			dtypes = [(field, data_dict[field].dtype) for field in data_dict]
			structured_array = np.core.records.fromarrays(arrays, dtype=dtypes)

			# Assign the structured array to the specific timeframe
			if symbol not in self.symbols_data:
				self.symbols_data[symbol] = {}

			self.symbols_data[symbol][tf_name] = structured_array

			# Initialize current_tick_index for each symbol and timeframe
			if symbol not in self.current_tick_index:
				self.current_tick_index[symbol] = {}
			# Find the first index where 'time' >= self.start_time
			times = structured_array['time']
			first_valid_index = np.searchsorted(times, self.start_time)
			self.current_tick_index[symbol][tf_name] = first_valid_index

			print(f"    Loaded data for {symbol} on timeframe {tf_name} with {len(df)} bars.")


	def load_timeseries_data(self):
		"""
		Load the timeseries data from EURUSD Parquet file of backtest tf - self.advance_timeframe
		"""
		# Load the timeseries data for EURUSD
		filename = f"EURUSD_{self.advance_timeframe}.parquet"
		filepath = os.path.join(self.data_dir, filename)

		# Read the Parquet file
		try:
			df = pd.read_parquet(filepath, columns=['time'])
		except Exception as e:
			print(f"Error reading {filename}: {e}.")
			raise e

		# Convert 'time' column to datetime objects
		df['time'] = pd.to_datetime(df['time'], errors='raise')

		# Remove rows with any NaT in 'time'
		if df['time'].isnull().any():
			print(f"Warning: Some 'time' entries could not be parsed in {filename}. They will be dropped.")
			df = df.dropna(subset=['time'])

		# Filter data to include only rows where 'time' >= self.start_time
		if self.start_time:
			initial_row_count = len(df)
			df = df[df['time'] >= self.start_time]
			filtered_row_count = len(df)
			print(f"    Filtered data for EURUSD on timeframe {self.advance_timeframe}: {filtered_row_count} out of {initial_row_count} bars retained (from {self.data_start_time} for strategy testing starting at: {self.start_time}).")
		else:
			print(f"    No start_time specified. Loading all data for EURUSD on timeframe {self.advance_timeframe}.")

		# Convert 'time' column to NumPy array of Python datetime objects
		time_array = df['time'].to_numpy(dtype='object')

		# Find Integer Index for self.end_time
		end_time_index = np.searchsorted(time_array, self.end_time)

		if end_time_index < len(time_array):
			self.end_time_index = end_time_index
			print(f"    Found end_time_index at position {self.end_time_index} for end_time {self.end_time}.")
		else:
			print(f"    No data available after {self.end_time}. Setting end_time_index to the last available index.")
			self.end_time_index = len(time_array) - 1

		self.time_array = time_array
		self.time_index = 0


	def get_timeframe_name(self, timeframe):
		"""
		Get the timeframe name from its value.

		Parameters:
			timeframe (int): The timeframe constant.

		Returns:
			str: The name of the timeframe.
		"""
		return TIMEFRAMES_REVERSE.get(timeframe, None)

	def copy_rates_from_pos(self, symbol, timeframe, pos, count):
		"""
		Simulate MT5.copy_rates_from_pos() function.

		Parameters:
			symbol (str): The trading symbol.
			timeframe (int): The timeframe constant.
			pos (int): Starting position (0 is the newest bar).
			count (int): Number of bars to copy.

		Returns:
			np.recarray or None: NumPy recarray of rates or None if error.
		"""
		tf_name = self.get_timeframe_name(timeframe)
		current_index = self.current_tick_index[symbol][tf_name]

		start = current_index - count - 1  # 1 extra bar to match live trading logic
		end = current_index  # In backtest we don't see the current bar

		# Return the rates from the specified position 
		return self.symbols_data[symbol][tf_name][start:end]

	def account_info(self):
		"""
		Simulate MT5.account_info() function.

		Returns:
			dict: Account information.
		"""
		# Convert account info to native Python types
		return self.account

	def positions_get(self, ticket=None):
		"""
		Simulate MT5.positions_get() function.

		Parameters:
			ticket (int, optional): Specific ticket number to retrieve.

		Returns:
			dict or list or None: Single position dict, list of positions, or None if not found.
		"""
		if ticket:
			position = self.open_positions.get(ticket, None)
			if position:
				return position  # Return single position dict
			else:
				return None
		else:
			positions = list(self.open_positions.values())
			return positions  # Return list of positions

	def symbol_info_tick(self, symbol):
		"""
		Simulate MT5.symbol_info_tick() function.

		Parameters:
			symbol (str): The trading symbol.

		Returns:
			dict or None: Tick information or None if error.
		"""
		bar_data = self.get_current_bar_data(symbol)

		tick_time = bar_data['time']

		tick = {
			'time': int(pd.Timestamp(tick_time).timestamp()),
			'bid': bar_data['bid'],
			'ask': bar_data['ask'],
		}
		return tick

	def symbol_info(self, symbol):
		"""
		Simulate MT5.symbol_info() function with caching.

		Parameters:
			symbol (str): The trading symbol.

		Returns:
			dict: Symbol information.
		"""
		#TODO: remove the if statements - cache should be proloaded 
		if not hasattr(self, 'symbol_info_cache'):
			self.symbol_info_cache = {}

		if symbol in self.symbol_info_cache:
			return self.symbol_info_cache[symbol]

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
			'pip': 10 ** -(digits - 1),
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
		self.symbol_info_cache[symbol] = info
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

	def history_deals_get(self, ticket):
		"""
		Simulate MT5.history_deals_get() function.

		Parameters:
			ticket (int): Specific ticket number to retrieve.

		Returns:
			dict or None: Deal dictionary or None if not found. # none is ommotied in backtest
		"""

		return self.closed_positions.get(ticket)

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
		comment = request.get('comment', '')
		sl = request.get('sl', 0.0)
		tp = request.get('tp', 0.0)
		magic = request.get('magic', 0)
		position_ticket = request.get('position', None)

		if action == TRADE_ACTION_DEAL:
			if position_ticket is None:
				# Open new position
				return self.execute_market_order(symbol, order_type, volume, price, comment, sl, tp, magic)		
			else:
				# Close existing position
				return self.close_position(position_ticket, price, reason='CLOSE')
		elif action == TRADE_ACTION_SLTP:
			# Modify order
			return self.modify_order(request)

		# We should not be here!
		request['retcode'] = TRADE_RETCODE_ERROR
		return request

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
		# Get symbol info
		symbol_info = self.symbol_info(symbol)
		point = symbol_info['point']

		# Calculate required margin (simplified)
		required_margin =  volume  * leverage_contract_ratio # (contract_size * volume ) / leverage

		#TODO: implement logic to exit backtest and return prameters
		if required_margin > self.account['free_margin']:
			print("Not enough free margin to open position.")
			print(f"Required margin: {required_margin}, Free margin: {self.account['free_margin']}")
			print(f"current open positions: {self.open_positions}")
			result = {}
			result['retcode'] = TRADE_RETCODE_ERROR
			return result

		# Update account margin
		self.account['margin'] += required_margin  # margin is update for open / close position only
		self.account['free_margin'] = self.account['equity'] - self.account['margin']

		# Create position
		position = {
			'ticket': self.next_ticket,
			'symbol': symbol,
			'type': order_type,  # BUY or SELL
			'volume': volume,
			'price_open': price,
			'sl': sl,
			'tp': tp,
			'time': self.current_time,
			'comment': comment,
			'magic': magic,
			'profit': 0.0,
			'contract_size': contract_size,
			'point': point,
			'margin': required_margin
		}
		self.open_positions[self.next_ticket] = position
		self.next_ticket += 1

		# Log the trade
		trade_log = position.copy()
		self.trade_logs.append(trade_log)

		# Return success
		result = position.copy()
		result['action'] = TRADE_ACTION_DEAL
		result['retcode'] = TRADE_RETCODE_DONE
		result['order'] = position['ticket']
		result['retcode'] = TRADE_ACTIONS['DONE']
		return result

	def modify_order(self, request):
		"""
		Modify an existing order (e.g., change SL/TP).

		Parameters:
			request (dict): Modification request containing 'position', 'sl', 'tp'.

		Returns:
			dict or None: Modification result or None if error.
		"""
		position_ticket = request.get('position')

		position = self.open_positions[position_ticket]
		position['sl'] = request['sl']
		position['tp'] = request['tp']

		self.open_positions[position_ticket] = position

		# Log the modification
		modification_log = {
			'ticket': position_ticket,
			'symbol': position['symbol'],
			'type': position['type'],
			'volume': position['volume'],
			'price': position['price_open'],
			'sl': position['sl'],
			'tp': position['tp'],
			'time': self.current_time,
			'comment': position['comment'],
			'magic': position['magic'],
			'action': 'MODIFY'
		}
		self.trade_logs.append(modification_log)
		# TODO: check if no more data is needed
		return {
			'retcode': TRADE_RETCODE_DONE
		}


	def get_current_price(self, symbol, order_type):
		"""
		Get the current price for a symbol based on current_time and advance_timeframe.

		Parameters:
			symbol (str): The trading symbol.
			order_type (int): The order type (ORDER_TYPE_BUY or ORDER_TYPE_SELL).

		Returns:
			float or None: Current price or None if error.
		"""
		tf_name = self.advance_timeframe  # Use the advance timeframe
		current_index = self.current_tick_index[symbol][tf_name]

		data_dict = self.symbols_data[symbol][tf_name]


		if order_type == ORDER_TYPE_BUY:
			price = data_dict['ask'][current_index]
		elif order_type == ORDER_TYPE_SELL:
			price = data_dict['bid'][current_index]
		else:
			price = None

		return price

	def advance_time_step(self, current_time_index):
		"""
		Advance the simulation time to the next data point.
		"""
		self.time_index = current_time_index

		# Advance time
		self.current_time = self.time_array[self.time_index]

		# Update current_tick_index for each symbol and timeframe
		for symbol, tfs in self.symbols_data.items():
			for tf_name, structured_array in tfs.items():
				current_index = self.current_tick_index[symbol][tf_name]
				times = structured_array['time']

				# Advance current_tick_index if next bar time <= current_time
				#TODO: think about optimization - "smart" search or advance....
				while times[current_index + 1] <= self.current_time:
					current_index += 1
					self.current_tick_index[symbol][tf_name] = current_index # only update if there was a change


	def simulate_target_hit_close(self):
		"""
		Check if any open positions have hit their SL or TP and close them accordingly.
		"""
		# TODO: optimize this function - njit?
		tickets_to_check = list(self.open_positions.keys())  # Create a list to avoid runtime errors
		for ticket in tickets_to_check:
			position = self.open_positions[ticket]
			symbol = position['symbol']
			order_type = position['type']
			sl = position['sl']
			tp = position['tp']

			# Get current bar data for the symbol
			bar_data = self.get_current_bar_data(symbol)

			high = bar_data['high']
			low = bar_data['low']

			# Determine if SL or TP was hit and close the position
			if order_type == ORDER_TYPE_BUY:
				if sl and low <= sl:
					self.close_position(ticket, sl, reason='SL')
				elif tp and high >= tp:
					self.close_position(ticket, tp, reason='TP')
			elif order_type == ORDER_TYPE_SELL:
				if sl and high >= sl:
					self.close_position(ticket, sl, reason='SL')
				elif tp and low <= tp:
					self.close_position(ticket, tp, reason='TP')

	def calculate_open_positions_profit(self):
		"""
		Calculate the profit for all open positions.
		Returns:
			float: The total profit for all open positions.
		"""
		total_profit = 0.0
		for position in self.open_positions.values():
			profit, _ = self.calculate_profit(position)
			total_profit += profit
		return total_profit


	def update_account_metrics(self, total_profit):
		"""
		Update account metrics based on profit.
		"""
		# no change in balance - only changes when closing positions
		# no change in margin and free_margin - only changes when opening / closing positions
		self.account['profit'] = total_profit
		self.account['equity'] = self.account['balance'] + self.account['profit']
		self.account['free_margin'] = self.account['equity'] - self.account['margin']

	def close_all_positions(self):
		"""
		Close all open positions.
		"""
		#TODO: check if I need the list
		for ticket in list(self.open_positions.keys()):
			self.close_position(ticket)

	def get_current_bar_data(self, symbol):
		"""
		Get the current bar data for the symbol at current_time in backtest tf.

		Parameters:
			symbol (str): The trading symbol.

		Returns:
			np.recarray or None: Bar data or None if not found.
		"""
		tf_name = self.advance_timeframe  # Use the advance timeframe
		current_index = self.current_tick_index[symbol][tf_name]

		# Extract current bar data as a record

		return self.symbols_data[symbol][tf_name][current_index]

	def get_swap_rate(self, symbol, order_type):
		"""
		Retrieve the swap rate for a given symbol and order type.
		Simplified swap calculation

		Parameters:
			symbol (str): The trading symbol.
			order_type (int): The order type (buy or sell).

		Returns:
			float: The swap rate.
		"""
		return -5  # Simplified swap rate calculation

	def calculate_profit(self, position):
		"""
		Calculate profit based on pips profit, pip value, volume, and swap.

		Parameters:
			position (dict): The position information.

		Returns:
			tuple: The calculated profit and the swap cost.
		"""
		symbol = position['symbol']
		order_type = position['type']
		volume = position['volume']
		entry_price = position['price_open']
		contract_size = position['contract_size']

		# Determine pip value
		pip = self.symbol_info(symbol)['pip']
		pip_value = contract_size * pip * volume

		#  Calculate profit in pips
		if order_type == ORDER_TYPE_BUY:
			current_price = self.get_current_price(symbol, ORDER_TYPE_SELL) # get the ask price
			pips_profit = (current_price - entry_price) / pip
		elif order_type == ORDER_TYPE_SELL:
			current_price = self.get_current_price(symbol, ORDER_TYPE_BUY) # get the bid price
			pips_profit = (entry_price - current_price) / pip

		# Calculate profit in base currency
		profit = pips_profit * pip_value

		# Adjust profit: convert to USD profit if needed
		quote_currency = symbol[-3:]
		if quote_currency != 'USD':
			# Initialize conversion rate
			conversion_rate = None

			# If USD is not the quote currency, adjust using the conversion rate
			usd_base_currencies = ['EUR', 'GBP', 'AUD', 'NZD', 'USD']
			usd_quote_currencies = ['JPY', 'CHF', 'CAD']
			if quote_currency in usd_base_currencies:
				# Convert to USD using the conversion rate
				conversion_rate = self.get_current_price(symbol=f"{quote_currency}USD", order_type=ORDER_TYPE_BUY)
			elif quote_currency in usd_quote_currencies:
				# Convert to USD using the reverse conversion rate
				conversion_rate = 1 / self.get_current_price(symbol=f"USD{quote_currency}", order_type=ORDER_TYPE_BUY)

			# Apply conversion rate if valid
			if conversion_rate and conversion_rate > 0:
				profit *= conversion_rate
			else:
				# TODO: Add to logging
				print(f"Warning: Could not retrieve conversion rate for {quote_currency}USD. Profit may be inaccurate.")

		# Simplified swap calculation:
		swap_rate = self.get_swap_rate(symbol, order_type)  # Retrieve swap rate for full lot
		days_in_trade = (self.current_time - position['time']).days
		swap_cost = days_in_trade * swap_rate * volume

		return profit, swap_cost


	def close_position(self, ticket, exit_price=None, reason='CLOSE'):
		"""
		Close a single position.

		Parameters:
			ticket (int): The ticket number of the position to close.
			exit_price (float): The price at which the position is closed.
			reason (str): The reason for closing ('CLOSE', 'SL', 'TP').
		"""
		position = self.open_positions[ticket]

		# Calculate profit
		profit , swap_cost = self.calculate_profit(position)

		total_profit = profit + swap_cost

		# Update account balance
		self.account['balance'] += total_profit
		self.account['margin'] -= position['margin'] # margin is update for open / close position only
		self.account['free_margin'] = self.account['equity'] - self.account['margin']


		# Move position to closed_positions
		closed_position = position.copy()
		closed_position['close_datetime'] = self.current_time
		closed_position['close_price'] = exit_price
		closed_position['profit'] = profit
		closed_position['reason'] = reason
		closed_position['swap'] = swap_cost
		self.closed_positions[ticket] = closed_position

		# Log the trade closure
		trade_log = {
			'ticket': ticket,
			'symbol': position['symbol'],
			'type': position['type'],
			'volume': position['volume'],
			'price': exit_price,
			'sl': position['sl'],
			'tp': position['tp'],
			'time': self.current_time,
			'comment': position['comment'],
			'magic': position['magic'],
			'profit': profit,
			'action': reason,
			'swap': closed_position['swap']
		}
		self.trade_logs.append(trade_log)

		# Remove from open_positions
		result =  self.open_positions.pop(ticket)
		result['retcode'] = TRADE_RETCODE_DONE
		result['order'] = ticket
		result['profit'] = total_profit
		print(f"Closed {position['type']}, no. {ticket} with profit: {total_profit}")
		return result

	def step_simulation(self, current_time_index):
		"""
		Advance the simulation time to the next data point.
		"""
		self.advance_time_step(current_time_index)

		# Execute trade logic (Check if hit SL or TP, update account)
		self.simulate_target_hit_close()

		# Log account status if a new hour has started
		if self.current_time.hour != self.previous_hour:
			total_profit = self.calculate_open_positions_profit()
			self.update_account_metrics(total_profit)
			self.log_account_status()
			self.previous_hour = self.current_time.hour
			

	def log_account_status(self):
		"""
		Log the account status at each simulated hour.
		"""
		#TODO: test throughly
		account_doc = {
			'datetime': self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
			'open_trades': len(self.open_positions),
			'balance': self.account['balance'],
			'equity': self.account['equity'],
			'profit': self.account['profit'],
			'margin': self.account['margin'],
			'free_margin': self.account['free_margin']
		}
		self.account_docs.append(account_doc)
		#TODO: delete once tested
		print(f"Account status logged at {self.current_time}: Balance: {self.account['balance']}, Equity: {self.account['equity']}, Open trades: {len(self.open_positions)}, profit: {self.account['profit']}")
		

	def export_logs(self):
		"""
		Export trade logs and account documentation to CSV files.
		"""
		# Export trade logs
		#TODO: test throughly
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
		#TODO: update to show profit and open trades
		#TODO: create 2 graphs - one for balance and equity, and one for profit, open trades and margin level
		#TODO: show grapfhs together? - need to correlate the time axis
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

		fig.add_trace(go.Scatter(
			x=account_df['datetime'],
			y=account_df['margin'],
			mode='lines',
			name='Margin'
		))

		fig.update_layout(
			title='Account Balance, Equity and Margin Over Time',
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

		# Show the graph
		fig.show()
		



	def time_current(self):
		"""
		Simulate MT5.time_current() function.

		Returns:
			datetime: Current time in the simulation.
		"""
		return self.current_time

	def shutdown(self):
		"""
		Simulate MT5.shutdown() function.
		"""
		# For backtesting, shutdown might reset the environment
		self.current_time = None
		self.open_positions.clear()
		self.closed_positions.clear()
		self.account = {
			'balance': initial_balance,
			'equity': initial_balance,
			'margin': 0.0, 
			'free_margin': initial_balance, # used only in backtest
			'profit': 0.0,
			'margin_level': 0.0 # used only in live trading - here for consistency
		}


# ----------------------------
# Exported Functions
# ----------------------------

# Global backtest instance - will support the exported functions
backtest = None  # Global backtest instance to be initialized later

def initialize_backtest(strategies):
	"""
	Initialize the global backtest instance with the provided strategies.

	Parameters:
		strategies (dict): Dictionary of strategy instances.
	"""
	global backtest
	backtest = MT5Backtest(strategies=strategies)
	main_params = {
		'start_time': backtest.current_time,
		'end_time': backtest.end_time,
		'time_step': backtest.time_step,
		'strategies': strategies,
		'symbols': backtest.symbols,
	}
	print(f"Backtest initialized with parameters: {main_params}")

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

def history_deals_get(ticket=None):
	"""
	Simulate MT5.history_deals_get() function.

	Parameters:
		ticket (int, optional): Specific ticket number to retrieve.

	Returns:
		dict or None: Deal dictionary or None if not found.
	"""
	return backtest.history_deals_get(ticket=ticket)

def log_account_status(account_info_dict, open_trades):
	"""
	Log the account status at each simulated hour.
	"""
	return backtest.log_account_status(account_info_dict, open_trades)

def last_error():
	"""
	Simulate MT5.last_error() function.

	Returns:
		tuple: (error_code, error_description)
	"""
	print("************************************************************************************************")
	print("################################################################################################")
	print(f"last_error() was called but it is not implemented in backtest mode.")
	print("################################################################################################")
	print("************************************************************************************************")

	# return backtest.last_error()

def time_current():
	"""
	Simulate MT5.time_current() function.

	Returns:
		datetime: Current time in the simulation.
	"""
	return backtest.time_current()

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

	from ..run_bot import on_minute

	# Initialize TradeHour and TimeBar with backtest
	trade_hour = TradeHour(backtest)
	time_bar = TimeBar(backtest)

	print(f"Starting backtest from {backtest.current_time} to {backtest.end_time}")
	try:
		#TODO: optimize - why for loop?
		for i in range(backtest.time_index, backtest.end_time_index):
			# Advance time by time_step and simulate trade logic
			backtest.step_simulation(i)

			# Call on_minute with current simulation time
			on_minute(strategies, trade_hour, time_bar, symbols, account_info_dict=None, BACKTEST_MODE=True)
	except Exception as e:
		print(f"Backtest stopped due to error: {e}")
		raise e
	finally:
		# Close all open positions at the end of backtest
		backtest.close_all_positions()
		# Log the final account status
		print("Backtest completed.")
		print(f"Final balance: {backtest.account['balance']}")
		print(f"Final equity: {backtest.account['equity']}")
		print(f"Total profit: {backtest.account['profit']}")

		# Export logs
		backtest.export_logs()

