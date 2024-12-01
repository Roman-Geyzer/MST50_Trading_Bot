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
import pandas as pd
from datetime import datetime, timedelta
import ta


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


# Global variable for required columns per symbol and timeframe
REQUIRED_COLUMNS = {}  # e.g., {'EURUSD': {'H1': set([...]), ...}, ...}

def initialize_required_indicator_columns(strategies):
    """
    Initialize the global REQUIRED_COLUMNS variable based on the required columns in the strategies.

    Parameters:
        strategies: A list of strategy instances that contain required_columns.
    """
    global REQUIRED_COLUMNS
    base_columns = ['open', 'high', 'low', 'close', 'ATR']
    
    for strategy in strategies.values():
        timeframes_with_required_columns = {strategy.str_timeframe: strategy.required_columns}
        timeframes_with_base_columns = {
            timeframe_to_string(strategy.higher_timeframe): base_columns,
            timeframe_to_string(strategy.lower_timeframe): base_columns,
            "M1": base_columns  # Add "M1" explicitly as a timeframe with base_columns
        }

        for symbol in strategy.symbols:
            if symbol not in REQUIRED_COLUMNS:
                REQUIRED_COLUMNS[symbol] = {}
                
            # Add required_columns for str_timeframe
            timeframe = strategy.str_timeframe
            if timeframe not in REQUIRED_COLUMNS[symbol]:
                REQUIRED_COLUMNS[symbol][timeframe] = set()
            REQUIRED_COLUMNS[symbol][timeframe].update(strategy.required_columns)

            # Add base_columns for other timeframes, including "M1"
            for timeframe, columns in timeframes_with_base_columns.items():
                if timeframe not in REQUIRED_COLUMNS[symbol]:
                    REQUIRED_COLUMNS[symbol][timeframe] = set()
                REQUIRED_COLUMNS[symbol][timeframe].update(columns)




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

def history_deals_get(ticket):
    return mt5_server.history_deals_get(ticket)

def copy_rates_from(symbol, timeframe, from_date, count):
    #not in use in the current implementation
    pass


#TODO: check this function in live trading
def copy_rates_from_pos(symbol, timeframe, pos, count):
    rates_list = mt5_server.copy_rates_from_pos(symbol, timeframe, pos, count + 1)  # collect 1 bar extra since the last bar is probably incomplete
    if rates_list is None:
        return None

    # Get the timestamp of the last bar in the rates_list
    last_bar_time = rates_list[-1]['time']  # get the time of the last bar
    if check_incomplete(last_bar_time):
        del rates_list[-1]  # remove the last bar from the list - it is incomplete (expected)

    rates_array = np.array([tuple(d.values()) for d in rates_list], dtype=dtype)

    # Convert rates_array to DataFrame for indicator calculations
    df = pd.DataFrame(rates_array)

    # Convert 'time' column from timestamp to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Get required columns for this symbol and timeframe
    timeframe_str = timeframe_to_string(timeframe)
    required_columns = REQUIRED_COLUMNS.get(symbol, {}).get(timeframe_str, set())

    if required_columns:
        df = calculate_indicators(df, required_columns)

        # Prepare extended dtype to include required columns
        extended_dtype = dtype.copy()
        for col in required_columns:
            if col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                    extended_dtype.append((col, 'float64'))
                elif df[col].dtype == 'bool':
                    extended_dtype.append((col, '?'))  # Boolean
                elif df[col].dtype == 'object':
                    max_len = df[col].astype(str).str.len().max()
                    extended_dtype.append((col, f'U{max_len}'))  # Unicode string
                else:
                    # Handle other data types as needed
                    pass

        # Ensure all extended_dtype fields are unique
        extended_dtype = list({dt[0]: dt for dt in extended_dtype}.values())

        # Reorder df columns to match extended_dtype
        df = df[[dt[0] for dt in extended_dtype]]

        # Fill NaN values with appropriate defaults
        for col, dt in zip(df.columns, extended_dtype):
            if dt[1] == 'float64':
                df[col] = df[col].fillna(np.nan)  # Assign the filled column back
            elif dt[1] == '?':
                df[col] = df[col].fillna(False)
            elif dt[1].startswith('U'):
                df[col] = df[col].fillna('')

        # Convert df to structured array
        
        rates_array = df.to_records(index=False)

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
    now = datetime.now() + timedelta(hours=server_time_hours_delta)
    if now.hour >= 24:
        now = now - timedelta(hours=24)
    return now


# SR calculation functions
def calculate_sr_levels(df, sr_params):
    """
    Calculate Support and Resistance (SR) levels and add them as new columns to the DataFrame.

    Parameters:
    - df: pandas DataFrame sorted by 'time' in ascending order with necessary price data and ATR
    - sr_params: dict containing SR calculation parameters

    Returns:
    - df: pandas DataFrame with SR indicator columns added
    """

    # Initialize SR columns with default values
    df['upper_sr'] = 0.0
    df['lower_sr'] = 0.0
    df['prev_upper_sr_level'] = 0.0
    df['prev_lower_sr_level'] = 0.0

    # Extract SR parameters
    period_for_sr = sr_params['period_for_sr']
    touches_for_sr = sr_params['touches_for_sr']
    slack_for_sr_atr_div = sr_params['slack_for_sr_atr_div']
    atr_rejection_multiplier = sr_params['atr_rejection_multiplier']
    min_height_of_sr_distance = sr_params['min_height_of_sr_distance']
    max_height_of_sr_distance = sr_params['max_height_of_sr_distance']

    # Iterate over the DataFrame to calculate SR levels
    for i in range(len(df)):
        if i < period_for_sr:
            # Not enough data to calculate SR levels
            continue

        # Define the window for SR calculation
        window_start = i - period_for_sr
        window_end = i  # Exclusive

        recent_rates = df.iloc[window_start:window_end]

        atr = df.at[i, 'ATR']
        if pd.isna(atr) or atr == 0:
            # Skip if ATR is not available
            continue

        uSlackForSR = atr / slack_for_sr_atr_div
        uRejectionFromSR = atr * atr_rejection_multiplier

        current_open = df.at[i, 'open']

        # Initialize HighSR and LowSR
        HighSR = current_open + min_height_of_sr_distance * uSlackForSR
        LowSR = current_open - min_height_of_sr_distance * uSlackForSR

        # LocalMax and LocalMin
        LocalMax = recent_rates['high'].max()
        LocalMin = recent_rates['low'].min()

        # Initialize LoopCounter
        LoopCounter = 0

        # Upper SR Level
        upper_sr_level = 0.0
        while LoopCounter < max_height_of_sr_distance:
            UpperSR = HighSR
            if count_touches(UpperSR, recent_rates, uRejectionFromSR,touches_for_sr, upper=True):
                upper_sr_level = UpperSR
                break
            else:
                HighSR += uSlackForSR
                LoopCounter += 1
                if HighSR > LocalMax:
                    upper_sr_level = 0
                    break

        # Reset LoopCounter for LowerSR
        LoopCounter = 0

        # Lower SR Level
        lower_sr_level = 0.0
        while LoopCounter < max_height_of_sr_distance:
            LowerSR = LowSR
            if count_touches(LowerSR, recent_rates, uRejectionFromSR,touches_for_sr, upper=False):
                lower_sr_level = LowerSR
                break
            else:
                LowSR -= uSlackForSR
                LoopCounter += 1
                if LowSR < LocalMin:
                    lower_sr_level = 0
                    break

        # Store SR levels in the DataFrame
        df.at[i, 'upper_sr'] = upper_sr_level
        df.at[i, 'lower_sr'] = lower_sr_level

        # Store previous SR levels
        if i > 0:
            df.at[i, 'prev_upper_sr_level'] = df.at[i-1, 'upper_sr']
            df.at[i, 'prev_lower_sr_level'] = df.at[i-1, 'lower_sr']
        else:
            df.at[i, 'prev_upper_sr_level'] = 0.0
            df.at[i, 'prev_lower_sr_level'] = 0.0

    return df

def count_touches(current_hline, recent_rates, uRejectionFromSR,touches_for_sr , upper=True):
    """
    Count the number of touches to the given SR level.

    Parameters:
        current_hline (float): The SR level to check.
        recent_rates (DataFrame): The recent rates to check.
        uRejectionFromSR (float): The rejection slack based on ATR.
        upper (bool): True if checking for upper SR, False for lower SR.

    Returns:
        bool - True if the number of touches is equal to the required number of touches for SR. (actual touches >= required touches - stop early)
    """
    counter = 0
    half_rejection = uRejectionFromSR / 2.0
    for idx in range(len(recent_rates) - 1):
        open_price = recent_rates['open'].iloc[idx]
        close_price = recent_rates['close'].iloc[idx]
        high_price = recent_rates['high'].iloc[idx]
        low_price = recent_rates['low'].iloc[idx]
        candle_size = abs(high_price - low_price)


        if upper:
            # Upper SR check
            if open_price < current_hline and close_price < current_hline:
                if high_price > current_hline or (
                    candle_size > uRejectionFromSR and (current_hline - high_price) < half_rejection
                ):
                    counter += 1
                    if counter == touches_for_sr:
                        return True
        else:
            # Lower SR check
            if open_price > current_hline and close_price > current_hline:
                if low_price < current_hline or (
                    candle_size > uRejectionFromSR and (low_price - current_hline) < half_rejection
                ):
                    counter += 1
                    if counter == touches_for_sr:
                        return True
    return False



# Helper functions
def calculate_indicators(df, required_columns):
    """
    Calculate technical indicators and add them as new columns to the DataFrame.

    Parameters:
        df: pandas DataFrame with OHLC data.
        required_columns: set of required columns to calculate.

    Returns:
        df: pandas DataFrame with new indicator columns added.
    """

    # Ensure the DataFrame is sorted by time in ascending order
    df = df.sort_values(by='time').reset_index(drop=True)

    # Map required_columns to indicators to calculate
    indicators_to_calculate = set()

    if 'RSI' in required_columns:
        indicators_to_calculate.add('RSI')

    if 'ATR' in required_columns:
        indicators_to_calculate.add('ATR')

    ma_periods = [int(col.split('_')[1]) for col in required_columns if col.startswith('MA_') and col.split('_')[1].isdigit()]
    if ma_periods:
        indicators_to_calculate.add('MA')

    bb_deviations = set()
    for col in required_columns:
        if col.startswith('BB') and not col.endswith('_Bool_Above') and not col.endswith('_Bool_Below'):
            deviation_str = col[2:].split('_')[0]
            if deviation_str.isdigit():
                deviation = int(deviation_str) / 10.0
                bb_deviations.add(deviation)
    if bb_deviations:
        indicators_to_calculate.add('BB')

    ga_windows = [int(col.split('_')[1]) for col in required_columns if col.startswith('GA_') and col.split('_')[1].isdigit()]
    if ga_windows:
        indicators_to_calculate.add('GA')

    if 'SR' in required_columns or 'Breakout' in required_columns or 'Fakeout' in required_columns:
        indicators_to_calculate.add('SR')

    # Now calculate the required indicators
    if 'RSI' in indicators_to_calculate:
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['RSI'] = rsi_indicator.rsi()

    if 'ATR' in indicators_to_calculate:
        atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['ATR'] = atr_indicator.average_true_range()

    if 'MA' in indicators_to_calculate:
        for period in ma_periods:
            ma_label = f'MA_{period}'
            ma_indicator = ta.trend.SMAIndicator(close=df['close'], window=period)
            df[ma_label] = ma_indicator.sma_indicator()

    if any(col.endswith('_comp') for col in required_columns):
        for ma_label in [f'MA_{period}' for period in ma_periods]:
            comp_label = f'{ma_label}_comp'
            if comp_label in required_columns and ma_label in df.columns:
                df[comp_label] = np.where(
                    df['close'] > df[ma_label],
                    'above',
                    np.where(df['close'] < df[ma_label], 'below', 'equal')
                )

    if 'BB' in indicators_to_calculate:
        for deviation in bb_deviations:
            label = f'BB{int(deviation * 10)}'
            bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=deviation)
            df[f'{label}_Upper'] = bollinger.bollinger_hband()
            df[f'{label}_Middle'] = bollinger.bollinger_mavg()
            df[f'{label}_Lower'] = bollinger.bollinger_lband()
            if f'{label}_Bool_Above' in required_columns:
                df[f'{label}_Bool_Above'] = df['close'] > df[f'{label}_Upper']
            if f'{label}_Bool_Below' in required_columns:
                df[f'{label}_Bool_Below'] = df['close'] < df[f'{label}_Lower']

    if 'GA' in indicators_to_calculate:
        df['Is_Green'] = df['close'] > df['open']
        for window in ga_windows:
            ga_label = f'GA_{window}'
            df[ga_label] = df['Is_Green'].rolling(window=window).sum() / window
        df.drop(['Is_Green'], axis=1, inplace=True)

    if 'SR' in indicators_to_calculate:
        #TODO: change from hardcoded to parameter
        # SR Parameters     
        SR_PARAMS = {
            'period_for_sr': 100,                # Lookback period for SR levels
            'touches_for_sr': 3,                 # Number of touches for SR levels
            'slack_for_sr_atr_div': 10.0,        # Slack for SR levels based on ATR
            'atr_rejection_multiplier': 1.0,     # ATR rejection multiplier for SR levels
            'max_distance_from_sr_atr': 2.0,     # Used in trading decisions
            'min_height_of_sr_distance': 3.0,    # Min height of SR distance - used in calculating SR levels
            'max_height_of_sr_distance': 200.0,  # Max height of SR distance - used in calculating SR levels
        }
        # populate SR levels in the DataFrame using calculate_sr_levels function
        df = calculate_sr_levels(df, SR_PARAMS)
    return df

def timeframe_to_string(timeframe):
    """
    Convert MT5 timeframe constants to string representation.

    Parameters:
        timeframe: MT5 timeframe constant.

    Returns:
        str: String representation of the timeframe (e.g., 'H1').
    """
    timeframe_mapping = {
        TIMEFRAMES['M1']: 'M1',
        TIMEFRAMES['M5']: 'M5',
        TIMEFRAMES['M15']: 'M15',
        TIMEFRAMES['M30']: 'M30',
        TIMEFRAMES['H1']: 'H1',
        TIMEFRAMES['H4']: 'H4',
        TIMEFRAMES['D1']: 'D1',
        TIMEFRAMES['W1']: 'W1',
    }
    return timeframe_mapping.get(timeframe, str(timeframe))