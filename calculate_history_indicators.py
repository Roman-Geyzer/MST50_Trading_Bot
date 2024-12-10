# calculate_history_indicators.py

"""
This script reads historical data files for currency pairs and timeframes,
calculates technical indicators and SR levels, and updates the data files with the new columns.
It utilizes Numba for JIT compilation and multiprocessing for parallel processing to optimize performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import ta  # Technical Analysis library
import time
import numba
from numba import njit, prange
from multiprocessing import Pool, cpu_count

drive = "/Volumes/TM"
folder = "historical_data"

# List of 28 major currency pairs
currency_pairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD',
    'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF',
    'CHFJPY'
]

# Timeframes
timeframes = [
    'M1',
    'M5',
    'M15',
    'M30',
    'H1',
    'H4',
    'D1',
    'W1',
]

# SR Parameters
# Lookback period for SR levels, number of touches,Slack for SR levels based on ATR  , ATR rejection multiplier
SR_configs = [
        (75,  3, 5, 0.5 , 'SR75_3_5_0.5'),
        (200, 3, 5, 0.5 , 'SR200_3_5_0.5'),
        (500, 3, 5, 0.5 , 'SR500_3_5_0.5'),
        (75,  4, 5, 0.5 , 'SR75_4_5_0.5'),
        (200, 4, 5, 0.5 , 'SR200_4_5_0.5'),
        (500, 4, 5, 0.5 , 'SR500_4_5_0.5'),
        (75,  5, 5, 0.5 , 'SR75_5_5_0.5'),
        (200, 5, 5, 0.5 , 'SR200_5_5_0.5'),
        (500, 5, 5, 0.5 , 'SR500_5_5_0.5'),
        (75,  3, 10, 0.5 , 'SR75_3_10_0.5'),
        (200, 3, 10, 0.5 , 'SR200_3_10_0.5'),
        (500, 3, 10, 0.5 , 'SR500_3_10_0.5'),
        (75,  4, 10, 0.5 , 'SR75_4_10_0.5'),
        (200, 4, 10, 0.5 , 'SR200_4_10_0.5'),
        (500, 4, 10, 0.5 , 'SR500_4_10_0.5'),
        (75,  5, 10, 0.5 , 'SR75_5_10_0.5'),
        (200, 5, 10, 0.5 , 'SR200_5_10_0.5'),
        (500, 5, 10, 0.5 , 'SR500_5_10_0.5'),
        (75,  3, 15, 0.5 , 'SR75_3_15_0.5'),
        (200, 3, 15, 0.5 , 'SR200_3_15_0.5'),
        (500, 3, 15, 0.5 , 'SR500_3_15_0.5'),
        (75,  4, 15, 0.5 , 'SR75_4_15_0.5'),
        (200, 4, 15, 0.5 , 'SR200_4_15_0.5'),
        (500, 4, 15, 0.5 , 'SR500_4_15_0.5'),
        (75,  5, 15, 0.5 , 'SR75_5_15_0.5'),
        (200, 5, 15, 0.5 , 'SR200_5_15_0.5'),
        (500, 5, 15, 0.5 , 'SR500_5_15_0.5'),
        (75,  3, 5, 1.0 , 'SR75_3_5_1.0'),
        (200, 3, 5, 1.0 , 'SR200_3_5_1.0'),
        (500, 3, 5, 1.0 , 'SR500_3_5_1.0'),
        (75,  4, 5, 1.0 , 'SR75_4_5_1.0'),
        (200, 4, 5, 1.0 , 'SR200_4_5_1.0'),
        (500, 4, 5, 1.0 , 'SR500_4_5_1.0'),
        (75,  5, 5, 1.0 , 'SR75_5_5_1.0'),
        (200, 5, 5, 1.0 , 'SR200_5_5_1.0'),
        (500, 5, 5, 1.0 , 'SR500_5_5_1.0'),
        (75,  3, 10, 1.0 , 'SR75_3_10_1.0'),
        (200, 3, 10, 1.0 , 'SR200_3_10_1.0'),
        (500, 3, 10, 1.0 , 'SR500_3_10_1.0'),
        (75,  3, 10, 1.0 , 'SR75_4_10_1.0'),
        (200, 4, 10, 1.0 , 'SR200_4_10_1.0'),
        (500, 4, 10, 1.0 , 'SR500_4_10_1.0'),
        (75,  5, 10, 1.0 , 'SR75_5_10_1.0'),
        (200, 5, 10, 1.0 , 'SR200_5_10_1.0'),
        (500, 5, 10, 1.0 , 'SR500_5_10_1.0'),
        (75,  3, 15, 1.0 , 'SR75_3_15_1.0'),
        (200, 3, 15, 1.0 , 'SR200_3_15_1.0'),
        (500, 3, 15, 1.0 , 'SR500_3_15_1.0'),
        (75,  4, 15, 1.0 , 'SR75_4_15_1.0'),
        (200, 4, 15, 1.0 , 'SR200_4_15_1.0'),
        (500, 4, 15, 1.0 , 'SR500_4_15_1.0'),
        (75,  5, 15, 1.0 , 'SR75_5_15_1.0'),
        (200, 5, 15, 1.0 , 'SR200_5_15_1.0'),
        (500, 5, 15, 1.0 , 'SR500_5_15_1.0'),
        (75,  3, 5, 1.5 , 'SR75_3_5_1.5'),
        (200, 3, 5, 1.5 , 'SR200_3_5_1.5'),
        (500,  3, 5, 1.5 , 'SR500_3_5_1.5'),
        (75,   4, 5, 1.5 , 'SR75_4_5_1.5'),
        (200,  4, 5, 1.5 , 'SR200_4_5_1.5'),
        (500,  4, 5, 1.5 , 'SR500_4_5_1.5'),
        (75,   5, 5, 1.5 , 'SR75_5_5_1.5'),
        (200,  5, 5, 1.5 , 'SR200_5_5_1.5'),
        (500,  5, 5, 1.5 , 'SR500_5_5_1.5'),
        (75,   3, 10, 1.5 , 'SR75_3_10_1.5'),
        (200,  3, 10, 1.5 , 'SR200_3_10_1.5'),
        (500,  3, 10, 1.5 , 'SR500_3_10_1.5'),
        (75,   4, 10, 1.5 , 'SR75_4_10_1.5'),
        (200,  4, 10, 1.5 , 'SR200_4_10_1.5'),
        (500,  4, 10, 1.5 , 'SR500_4_10_1.5'),
        (75,   5, 10, 1.5 , 'SR75_5_10_1.5'),
        (200,  5, 10, 1.5 , 'SR200_5_10_1.5'),
        (500,  5, 10, 1.5 , 'SR500_5_10_1.5'),
        (75,   3, 15, 1.5 , 'SR75_3_15_1.5'),
        (200,  3, 15, 1.5 , 'SR200_3_15_1.5'),
        (500,  3, 15, 1.5 , 'SR500_3_15_1.5'),
        (75,   4, 15, 1.5 , 'SR75_4_15_1.5'),
        (200,  4, 15, 1.5 , 'SR200_4_15_1.5'),
        (500,  4, 15, 1.5 , 'SR500_4_15_1.5'),
        (75,   5, 15, 1.5 , 'SR75_5_15_1.5'),
        (200,  5, 15, 1.5 , 'SR200_5_15_1.5'),
        (500,  5, 15, 1.5 , 'SR500_5_15_1.5'),
    ]


fixed_SR_params = {
    'min_height_of_sr_distance': 3.0,    # Min height of SR distance - used in calculating SR levels
    'max_height_of_sr_distance': 60.0,   # Max height of SR distance - used in calculating SR levels
}

def calculate_indicators(df, pip):
    """
    Calculate technical indicators and add them as new columns to the DataFrame.
    """
    # Ensure the DataFrame is sorted by time in ascending order
    df = df.sort_values(by='time').reset_index(drop=True)

    # Check if necessary columns exist
    required_columns = ['open', 'high', 'low', 'close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert to float32 to save memory
    df['open'] = df['open'].astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')

    # 1. Calculate RSI's
    RSIs = [2,7, 14, 21, 50]
    for rsi_period in RSIs:
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=rsi_period)
        df[f'RSI_{rsi_period}'] = rsi_indicator.rsi().astype('float32')

    # 2. Calculate ATR (Period 14)
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr_indicator.average_true_range().astype('float32')

    # 3. Calculate Bollinger Bands with different deviations
    bollinger_settings = [
        (15, 1.5, 'BB15_1.5'),
        (15, 2.0, 'BB15_2.0'),
        (15, 2.5, 'BB15_2.5'),
        (20, 1.5, 'BB20_1.5'),
        (20, 2.0, 'BB20_2.0'),
        (20, 2.5, 'BB20_2.5'),
        (25, 1.5, 'BB25_1.5'),
        (25, 2.0, 'BB25_2.0'),
        (25, 2.5, 'BB25_2.5'),
    ]

    for period, deviation, label in bollinger_settings:
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=period, window_dev=deviation)
        df[f'{label}_Upper'] = bollinger.bollinger_hband().astype('float32')
        df[f'{label}_Middle'] = bollinger.bollinger_mavg().astype('float32')
        df[f'{label}_Lower'] = bollinger.bollinger_lband().astype('float32')
        # Boolean flags for Close above Upper Band and below Lower Band
        df[f'{label}_Bool_Above'] = (df['close'] > df[f'{label}_Upper'])
        df[f'{label}_Bool_Below'] = (df['close'] < df[f'{label}_Lower'])

    # 4. Calculate Moving Averages
    moving_averages = {
        'MA_7': 7,      # Short-term
        'MA_21': 21,    # Medium-term
        'MA_50': 50,    # Long-term
        'MA_200': 200   # Very Long-term
    }

    for ma_label, period in moving_averages.items():
        ma_indicator = ta.trend.SMAIndicator(close=df['close'], window=period)
        df[ma_label] = ma_indicator.sma_indicator().astype('float32')

    # 5. Calculate Green-Red Candle Ratios
    # Use Numba-accelerated function
    df['Is_Green'] = (df['close'] > df['open']).astype('int8')
    green_red_settings = [
        (50, 'GA_50'),
        (100, 'GA_100'),
        (200, 'GA_200'),
        (500, 'GA_500')
    ]

    for window, column_name in green_red_settings:
        df[column_name] = rolling_sum_numba(df['Is_Green'].values, window) / window

    # Drop intermediate columns if not needed
    df.drop(['Is_Green'], axis=1, inplace=True)

    # 6. Compare Price with Moving Averages
    for ma_label in ['MA_7', 'MA_21', 'MA_50']:
        df[f'{ma_label}_comp'] = np.where(
            df['close'] > df[ma_label],
            'above',
            np.where(df['close'] < df[ma_label], 'below', 'equal')
        )

    # 7. Calculate bid and ask
    df['bid'] = df['open'] - df['spread'] * pip / 2
    df['ask'] = df['open'] + df['spread'] * pip / 2

    return df

@njit(parallel=True)
def rolling_sum_numba(data, window):
    """
    Compute rolling sum using Numba for acceleration.
    """
    result = np.empty(len(data), dtype=np.float32)
    cumulative_sum = 0.0
    for i in range(len(data)):
        cumulative_sum += data[i]
        if i >= window:
            cumulative_sum -= data[i - window]
            result[i] = cumulative_sum
        elif i == window - 1:
            result[i] = cumulative_sum
        else:
            result[i] = np.nan
    # Prepend NaNs for the initial values where the window is not full
    for i in range(window - 1):
        result[i] = np.nan
    return result

def calculate_sr_levels(df, sr_params, upper_sr_col, lower_sr_col):
    """
    Calculate Support and Resistance (SR) levels and add them as new columns to the DataFrame.
    Parameters:
        df (pd.DataFrame): DataFrame containing historical price data.
        sr_params (dict): Dictionary containing SR calculation parameters.
        upper_sr_col (str): Column name for the upper SR levels.
        lower_sr_col (str): Column name for the lower SR levels.
    """
    # Initialize SR columns with default values
    df['upper_sr'] = 0.0
    df['lower_sr'] = 0.0

    # Extract SR parameters
    period_for_sr = sr_params['period_for_sr']
    touches_for_sr = sr_params['touches_for_sr']
    slack_for_sr_atr_div = sr_params['slack_for_sr_atr_div']
    atr_rejection_multiplier = sr_params['atr_rejection_multiplier']
    min_height_of_sr_distance = sr_params['min_height_of_sr_distance']
    max_height_of_sr_distance = sr_params['max_height_of_sr_distance']

    # Prepare data arrays for Numba function
    open_prices = df['open'].values.astype(np.float32)
    high_prices = df['high'].values.astype(np.float32)
    low_prices = df['low'].values.astype(np.float32)
    close_prices = df['close'].values.astype(np.float32)
    atr_values = df['ATR'].values.astype(np.float32)

    upper_sr_array = np.zeros(len(df), dtype=np.float32)
    lower_sr_array = np.zeros(len(df), dtype=np.float32)

    # Call Numba-accelerated function
    calculate_sr_levels_numba(
        open_prices, high_prices, low_prices, close_prices, atr_values,
        upper_sr_array, lower_sr_array,
        period_for_sr, touches_for_sr, slack_for_sr_atr_div,
        atr_rejection_multiplier, min_height_of_sr_distance, max_height_of_sr_distance
    )

    df[upper_sr_col] = upper_sr_array
    df[lower_sr_col] = lower_sr_array

    return df

@njit(parallel=True)
def calculate_sr_levels_numba(
    open_prices, high_prices, low_prices, close_prices, atr_values,
    upper_sr_array, lower_sr_array,
    period_for_sr, touches_for_sr, slack_for_sr_atr_div,
    atr_rejection_multiplier, min_height_of_sr_distance, max_height_of_sr_distance
):
    """
    Numba-accelerated function to calculate SR levels.
    """
    n = len(open_prices)
    for i in prange(period_for_sr + 1, n):
        atr = atr_values[i]
        if atr == 0 or np.isnan(atr):
            continue

        uSlackForSR = atr / slack_for_sr_atr_div
        uRejectionFromSR = atr * atr_rejection_multiplier

        current_open = open_prices[i]

        # Initialize HighSR and LowSR
        HighSR = current_open + min_height_of_sr_distance * uSlackForSR
        LowSR = current_open - min_height_of_sr_distance * uSlackForSR

        # LocalMax and LocalMin in the recent window
        recent_highs = high_prices[i - period_for_sr:i]
        recent_lows = low_prices[i - period_for_sr:i]
        LocalMax = np.nanmax(recent_highs)
        LocalMin = np.nanmin(recent_lows)

        # Upper SR Level
        LoopCounter = 0
        upper_sr_level = 0.0
        while LoopCounter < max_height_of_sr_distance:
            UpperSR = HighSR
            if count_touches_numba(
                UpperSR, open_prices, high_prices, low_prices, close_prices, uRejectionFromSR,
                touches_for_sr, i - period_for_sr, i, upper=True
            ):
                upper_sr_level = UpperSR
                break
            else:
                HighSR += uSlackForSR
                LoopCounter += 1
                if HighSR > LocalMax:
                    upper_sr_level = 0
                    break

        # Lower SR Level
        LoopCounter = 0
        lower_sr_level = 0.0
        while LoopCounter < max_height_of_sr_distance:
            LowerSR = LowSR
            if count_touches_numba(
                LowerSR, open_prices, high_prices, low_prices, close_prices, uRejectionFromSR,
                touches_for_sr, i - period_for_sr, i, upper=False
            ):
                lower_sr_level = LowerSR
                break
            else:
                LowSR -= uSlackForSR
                LoopCounter += 1
                if LowSR < LocalMin:
                    lower_sr_level = 0
                    break

        upper_sr_array[i] = upper_sr_level
        lower_sr_array[i] = lower_sr_level

@njit
def count_touches_numba(
    current_hline, open_prices, high_prices, low_prices, close_prices, uRejectionFromSR,
    touches_for_sr, start_idx, end_idx, upper=True
):
    """
    Numba-accelerated function to count the number of touches to the given SR level.
    """
    counter = 0
    half_rejection = uRejectionFromSR / 2.0

    for idx in range(start_idx, end_idx - 1):
        open_price = open_prices[idx]
        close_price = close_prices[idx]
        high_price = high_prices[idx]
        low_price = low_prices[idx]
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

def process_symbol_timeframe(args):
    """
    Function to process a single symbol and timeframe. Designed for multiprocessing.
    """
    symbol, tf_name, output_dir = args
    print(f"Processing symbol: {symbol}, Timeframe: {tf_name}")
    print(f"    Output directory: {output_dir}")
    print(f"time is {datetime.now()}")
    if 'JPY' in symbol:
        pip_digits = 2
    else:
        pip_digits = 4
    pip = 10 ** -pip_digits

    # Define file path
    filename = f"{symbol}_{tf_name}.parquet"
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        print(f"    Data file {filepath} does not exist.")
        return

    try:
        df = pd.read_parquet(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by='time', inplace=True)
        df.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"    Error reading data file: {e}")
        return

    # Check if indicators are missing
    indicators_missing = False
    required_columns = ['RSI', 'ATR', 'BB15_Upper', 'MA_7', 'upper_sr']
    for col in required_columns:
        if col not in df.columns or df[col].isna().any():
            indicators_missing = True
            break

    if not indicators_missing:
        print("    Indicators already calculated for all data.")
        return

    # Calculate indicators
    try:
        df = calculate_indicators(df, pip)
    except Exception as e:
        print(f"    Error calculating indicators: {e}")
        return

    # Calculate SR levels
        # Iterate over each SR configuration
    for config in SR_configs:
        period_for_sr, touches_for_sr, slack_for_sr_atr_div, atr_rejection_multiplier, config_id = config
        
        # Update SR_PARAMS with current configuration
        SR_PARAMS = {
            'period_for_sr': period_for_sr,
            'touches_for_sr': touches_for_sr,
            'slack_for_sr_atr_div': slack_for_sr_atr_div,
            'atr_rejection_multiplier': atr_rejection_multiplier,
        }
        SR_PARAMS.update(fixed_SR_params)
        
        print(f"Calculating SR levels for configuration: {config_id}")
        
        try:
            # Calculate SR levels on a copy to avoid overwriting
            df = calculate_sr_levels(df.copy(), SR_PARAMS, f"upper_{config_id}", f"lower_{config_id}")
            print(f"    Calculated SR levels for {config_id}")
        except Exception as e:
            print(f"    Error calculating SR levels for {config_id}: {e}")
            continue  # Skip to the next configuration

        # Handle potential NaN values resulting from SR calculation
        df[[f"upper_{config_id}", f"lower_{config_id}"]] = df[[f"upper_{config_id}", f"lower_{config_id}"]].fillna(0)
        

    # Save updated data to Parquet
    try:
        df.to_parquet(filepath, index=False)
        print(f"    Updated data with indicators saved to {filepath}")
    except Exception as e:
        print(f"    Error saving data to Parquet: {e}")

def calculate_indicators_for_files():
    output_dir = os.path.join(drive, folder)
    if not os.path.exists(output_dir):
        print(f"Data directory {output_dir} does not exist.")
        return

    # Prepare arguments for multiprocessing
    tasks = []
    for symbol in currency_pairs:
        for tf_name in timeframes:
            tasks.append((symbol, tf_name, output_dir))

    # Use multiprocessing Pool
    num_processes = max(cpu_count() - 1, 1)  # Leave one core free
    with Pool(processes=num_processes) as pool:
        pool.map(process_symbol_timeframe, tasks)

if __name__ == "__main__":
    calculate_indicators_for_files()