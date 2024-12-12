# calculate_history_indicators.py

"""
This script reads historical data files for currency pairs and timeframes,
calculates technical indicators, SR levels, and candle patterns, and updates the data files with the new columns.
It utilizes Numba for JIT compilation and multiprocessing for parallel processing to optimize performance.

Logic:
- Split into 3 groups: Indicators, SR, Candle Patterns.
- If no data for a column in the group: calculate from start.
- If partial data missing: find first incomplete index and recalculate only from that index - 510.
- If full data: no recalculation.
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

MARUBOZU = 3
BULLISH_MARUBOZU = 4
BEARISH_MARUBOZU = 5
SAME_COLOR = 6
HH = 7
LL = 8
HHHC = 9
LLLC = 10

comparison_mapping = {
    'Marubozu': MARUBOZU,
    'Bullish_Marubozu': BULLISH_MARUBOZU,
    'Bearish_Marubozu': BEARISH_MARUBOZU,
    'Same_Color': SAME_COLOR,
    'HH': HH,
    'LL': LL,
    'HHHC': HHHC,
    'LLLC': LLLC,
}



def count_consecutive(df: pd.DataFrame, comparison_type: str) -> np.ndarray:
    n = len(df)
    counts = np.empty(n, dtype=np.int32)

    if n == 0:
        return counts

    counts[0] = 1  # The first row has a count of 1
    current_count = 1

    for i in range(1, n):
        condition = False

        if comparison_type == MARUBOZU:
            if df.at[i, 'Marubozu'] and df.at[i, 'canal_color'] == df.at[i - 1, 'candle_color']:
                condition = True

        elif comparison_type == BULLISH_MARUBOZU:
            if df.at[i, 'Bullish_Marubozu']:
                condition = True

        elif comparison_type == BEARISH_MARUBOZU:
            if df.at[i, 'Bearish_Marubozu']:
                condition = True

        elif comparison_type == SAME_COLOR:
            if df.at[i, 'candle_color'] == df.at[i - 1, 'candle_color']:
                condition = True

        elif comparison_type == HH:
            if df.at[i, 'high'] > df.at[i - 1, 'high']:
                condition = True

        elif comparison_type == LL:
            if df.at[i, 'low'] < df.at[i - 1, 'low']:
                condition = True

        elif comparison_type == HHHC:
            if df.at[i, 'HHHC']:
                condition = True

        elif comparison_type == LLLC:
            if df.at[i, 'LLLC']:
                condition = True

        if condition:
            current_count += 1
        else:
            current_count = 1

        counts[i] = current_count

    return counts

@njit
def compute_candle_measures(open_arr, high_arr, low_arr, close_arr, color_arr):
    n = len(open_arr)
    body_size = np.abs(close_arr - open_arr)
    candle_size = np.abs(high_arr - low_arr)
    
    upper_wik_size = np.empty(n, dtype=np.float32)
    lower_wik_size = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        if color_arr[i] == -1:  # Red candle
            upper_wik_size[i] = high_arr[i] - open_arr[i]
            lower_wik_size[i] = close_arr[i] - low_arr[i]
        else:  # Green or Doji
            upper_wik_size[i] = high_arr[i] - close_arr[i]
            lower_wik_size[i] = open_arr[i] - low_arr[i]
    
    upper_wik_ratio = np.where(upper_wik_size == 0, 1000, body_size / upper_wik_size)
    lower_wik_ratio = np.where(lower_wik_size == 0, 1000, body_size / lower_wik_size)
    sum_wik = upper_wik_size + lower_wik_size
    wik_ratio = np.where(sum_wik == 0, 10, body_size / sum_wik)

    return body_size, candle_size, upper_wik_size, lower_wik_size, upper_wik_ratio, lower_wik_ratio, wik_ratio

#Don't use at the moment
def populate_candles_measures(df):
    df['open'] = df['open'].astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')
    df['candle_color'] = df['candle_color'].astype('int8')

  

    body_size = np.abs(df['close'] - df['open'])
    candle_size = np.abs(df['high'] - df['low'])
    upper_wik_size = np.where(df['candle_color'] == -1, df['high'] - df['open'], df['high'] - df['close'])
    lower_wik_size = np.where(df['candle_color'] == -1, df['close'] - df['low'], df['open'] - df['low'])
    upper_wik_ratio = np.where(upper_wik_size == 0, 1000, body_size / upper_wik_size)
    lower_wik_ratio = np.where(lower_wik_size == 0, 1000, body_size / lower_wik_size)
    sum_wik = upper_wik_size + lower_wik_size
    wik_ratio = np.where(sum_wik == 0, 10, body_size / sum_wik)

    new_cols = pd.DataFrame({
        'Body_Size': body_size.astype('float32'),
        'Candle_Size': candle_size.astype('float32'),
        'Upper_Wik_Size': upper_wik_size.astype('float32'),
        'Lower_Wik_Size': lower_wik_size.astype('float32'),
        'Upper_wik_ratio': upper_wik_ratio.astype('float32'),
        'Lower_wik_ratio': lower_wik_ratio.astype('float32'),
        'wik_ratio': wik_ratio.astype('float32')
    }, index=df.index)

    df = pd.concat([df, new_cols], axis=1)
    return df

def calculate_patterns(df):
    # Parameters
    doji_ratio = 0.1
    upper_quarter_threshold = 0.75
    lower_quarter_threshold = 0.25
    wik_ratio_threshold = 0.25
    marubozu_threshold = 2.5

    # Basic candle color
    df['candle_color'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0)).astype(np.int8)

    # Compute candle measures (assuming compute_candle_measures is defined elsewhere)
    body_size, candle_size, upper_wik_size, lower_wik_size, upper_wik_ratio, lower_wik_ratio, wik_ratio = \
        compute_candle_measures(df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['candle_color'].values)

    # Direct column assignments
    df['Body_Size'] = body_size
    df['Candle_Size'] = candle_size
    df['Upper_Wik_Size'] = upper_wik_size
    df['Lower_Wik_Size'] = lower_wik_size
    df['Upper_wik_ratio'] = upper_wik_ratio
    df['Lower_wik_ratio'] = lower_wik_ratio
    df['wik_ratio'] = wik_ratio

    # Basic patterns
    df['HH'] = (df['high'] > df['high'].shift(1))
    df['LL'] = (df['low'] < df['low'].shift(1))
    df['HHHC'] = df['HH'] & (df['close'] > df['close'].shift(1))
    df['LLLC'] = df['LL'] & (df['close'] < df['close'].shift(1))
    df['Same_Color'] = df['candle_color'] == df['candle_color'].shift(1)

    # Calculate Doji
    body_center = (df['open'] + df['close']) / 2
    relative_pos = (body_center - df['low']) / df['Candle_Size']
    df['Doji'] = (df['Body_Size'] < df['Candle_Size'] * doji_ratio) & \
                 (relative_pos >= lower_quarter_threshold) & (relative_pos <= upper_quarter_threshold)

    # Marubozu patterns
    df['Marubozu'] = ((np.maximum(df['Upper_wik_ratio'], df['Lower_wik_ratio']) >= marubozu_threshold) &
                      (df['wik_ratio'] > 1.75) &
                      (df['candle_color'] != 0))
    df['Bullish_Marubozu'] = df['Marubozu'] & (df['candle_color'] == 1)
    df['Bearish_Marubozu'] = df['Marubozu'] & (df['candle_color'] == -1)

    # Hammer & Inverted Hammer
    df['Hammer'] = (df['wik_ratio'] <= wik_ratio_threshold) & \
                   ((df['Upper_Wik_Size'] == 0) | ((df['Lower_Wik_Size'] / df['Upper_Wik_Size']) > 2))
    df['Inverted_Hammer'] = (df['wik_ratio'] <= wik_ratio_threshold) & \
                            ((df['Lower_Wik_Size'] == 0) | ((df['Upper_Wik_Size'] / df['Lower_Wik_Size']) > 2))

    # Shadows & Doji shadows
    df['upper_shadow'] = df['Upper_Wik_Size'] > df['Body_Size'] * wik_ratio_threshold
    df['lower_shadow'] = df['Lower_Wik_Size'] > df['Body_Size'] * wik_ratio_threshold
    df['upper_shadow_doji'] = df['upper_shadow'] & df['Doji']
    df['lower_shadow_doji'] = df['lower_shadow'] & df['Doji']

    # Inside/Outside Bar
    df['Outside_Bar'] = df['HH'].shift(1) & df['LL'].shift(1)
    df['Inside_Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))

    # Engulfing patterns
    df['Engulf'] = ((df['candle_color'] != df['candle_color'].shift(1)) &
                    (((df['candle_color'] == 1) & (df['open'] <= df['close'].shift(1)) & (df['close'] >= df['open'].shift(1))) |
                     ((df['candle_color'] == -1) & (df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1)))))
    df['Bullish_Engulfing'] = (df['candle_color'].shift(1) == -1) & (df['candle_color'] == 1) & df['Engulf']
    df['Bearish_Engulfing'] = (df['candle_color'].shift(1) == 1) & (df['candle_color'] == -1) & df['Engulf']

    df['Marubozu_Doji'] = df['Marubozu'] & df['Doji'].shift(-1)

    # Harami patterns
    df['Bullish_Harami'] = (df['candle_color'].shift(1) == -1) & (df['candle_color'] == 1) & df['Inside_Bar']
    df['Bearish_Harami'] = (df['candle_color'].shift(1) == 1) & (df['candle_color'] == -1) & df['Inside_Bar']

    # HHHL & LHLL
    df['HHHL'] = (df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['LHLL'] = (df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1))

    # Kangaroo Tail
    df['Kangaroo_Tail'] = ((df['candle_color'].shift(1) == 1) & df['LHLL'].shift(2) & df['HHHL'].shift(1)) | \
                          ((df['candle_color'].shift(1) == -1) & df['HHHL'].shift(2) & df['LHLL'].shift(1))
    df['Kangaroo_Tail_Bullish'] = df['Kangaroo_Tail'] & (df['candle_color'] == 1)
    df['Kangaroo_Tail_Bearish'] = df['Kangaroo_Tail'] & (df['candle_color'] == -1)

    # Partial Kangaroo
    df['Partial_Kangaroo_Bullish'] = ((df['low'].shift(2) < df['low'].shift(3)) & (df['low'].shift(2) < df['low'].shift(1)))
    df['Partial_Kangaroo_Bearish'] = ((df['high'].shift(2) > df['high'].shift(3)) & (df['high'].shift(2) > df['high'].shift(1)))

    # Morning/Evening Star
    df['Morning_Star'] = (df['candle_color'].shift(2) == -1) & (df['Doji'].shift(1)) & (df['candle_color'] == 1)
    df['Evening_Star'] = (df['candle_color'].shift(2) == 1) & (df['Doji'].shift(1)) & (df['candle_color'] == -1)

    # Three White Soldiers / Black Crows
    df['Three_White_Soldiers'] = (df['candle_color'].shift(2) == 1) & (df['candle_color'].shift(1) == 1) & (df['candle_color'] == 1)
    df['Three_Black_Crows'] = (df['candle_color'].shift(2) == -1) & (df['candle_color'].shift(1) == -1) & (df['candle_color'] == -1)
    df['Three_White_Soldiers_Doji'] = df['Three_White_Soldiers'] & df['Doji'].shift(-1)
    df['Three_Black_Crows_Doji'] = df['Three_Black_Crows'] & df['Doji'].shift(-1)

    # Kicker & Kanazawa
    df['Kicker'] = (df['candle_color'].shift(1) == 1) & (df['candle_color'] == -1)
    df['Kanazawa'] = (df['candle_color'].shift(1) == -1) & (df['candle_color'] == 1)
    df['Kicker_Doji'] = df['Kicker'] & df['Doji'].shift(-1)
    df['Kanazawa_Doji'] = df['Kanazawa'] & df['Doji'].shift(-1)

    # Harami Doji
    df['Bullish_Harami_Doji'] = (df['Bearish_Harami']) & df['Doji'].shift(-1)
    df['Bearish_Harami_Doji'] = (df['Bullish_Harami']) & df['Doji'].shift(-1)

    # Inside Breakout
    df['Inside_Breakout_Bullish'] = (df['Inside_Bar'].shift(2)) & (df['close'].shift(1) > df['high'].shift(3))
    df['Inside_Breakout_Bearish'] = (df['Inside_Bar'].shift(2)) & (df['close'].shift(1) < df['low'].shift(3))

    # Count consecutive occurrences for key patterns
    df['Same_Color_Count'] = count_consecutive(df, 'Same_Color')
    df['Marubozu_Count'] = count_consecutive(df, 'Marubozu')
    df['Bullish_Marubozu_Count'] = count_consecutive(df, 'Bullish_Marubozu')
    df['Bearish_Marubozu_Count'] = count_consecutive(df, 'Bearish_Marubozu')
    df['HH_Count'] = count_consecutive(df, 'HH')
    df['LL_Count'] = count_consecutive(df, 'LL')
    df['HHHC_Count'] = count_consecutive(df, 'HHHC')
    df['LLLC_Count'] = count_consecutive(df, 'LLLC')

    # Convert boolean columns to bool type
    pattern_bool_cols = [c for c in df.columns if df[c].dtype == bool or df[c].dtype == np.bool_]
    for c in pattern_bool_cols:
        df[c] = df[c].astype(bool)

    # Convert pattern columns to int8 for all count columns
    pattern_count_cols = [c for c in df.columns if c.endswith('_Count')]
    for c in pattern_count_cols:
        # Fill NaN with 0 before casting
        df[c] = df[c].fillna(0).astype(np.int8)

    return df

# Define the parameter ranges
# Lookback period for SR levels, number of touches,Slack for SR -  ATR devider , ATR rejection multiplier
period = [75, 200, 500]
touches = [3, 4, 5]
slack_div = [5, 10, 15]
rejection_multi = [0.5, 1.0, 1.5]

# Generate SR_configs using list comprehension
SR_configs = [
    (l, t, s, r, f'SR{l}_{t}_{s}_{r}')
    for r in rejection_multi
    for t in slack_div
    for s in touches
    for l in period
]

fixed_SR_params = {
    'min_height_of_sr_distance': 3.0,
    'max_height_of_sr_distance': 60.0,
}


def get_indicator_columns():
    indicator_cols = ['open', 'high', 'low', 'close', 'spread', 'time', 'bid', 'ask', 'ATR']
    RSIs = [2,7,14,21,50]
    for r in RSIs:
        indicator_cols.append(f'RSI_{r}')
    bb_settings = [
        (15, 1.5, 'BB15_1.5'), (15, 2.0, 'BB15_2.0'), (15, 2.5, 'BB15_2.5'),
        (20, 1.5, 'BB20_1.5'), (20, 2.0, 'BB20_2.0'), (20, 2.5, 'BB20_2.5'),
        (25, 1.5, 'BB25_1.5'), (25, 2.0, 'BB25_2.0'), (25, 2.5, 'BB25_2.5'),
    ]
    for period, deviation, label in bb_settings:
        indicator_cols.extend([f'{label}_Upper', f'{label}_Middle', f'{label}_Lower', f'{label}_Bool_Above', f'{label}_Bool_Below'])

    for p in [7,21,50,200]:
        indicator_cols.append(f'MA_{p}')
        if p in [7,21,50]:
            indicator_cols.append(f'MA_{p}_comp')
    for w in [50,100,200,500]:
        indicator_cols.append(f'GA_{w}')
    return list(set(indicator_cols))


def get_sr_columns():
    sr_cols = []
    for _, _, _, _, config_id in SR_configs:
        sr_cols.append(f'upper_{config_id}')
        sr_cols.append(f'lower_{config_id}')
    return list(set(sr_cols))


def get_pattern_columns():
    pattern_cols = [
        'candle_color','Body_Size','Candle_Size','Upper_Wik_Size','Lower_Wik_Size','Upper_wik_ratio','Lower_wik_ratio','wik_ratio',
        'Doji','Marubozu','Bullish_Marubozu','Bearish_Marubozu','Marubozu_Count','Bullish_Marubozu_Count','Bearish_Marubozu_Count',
        'Same_Color','Same_Color_Count','HH','LL','HHHC','LLLC','HH_Count','LL_Count','HHHC_Count','LLLC_Count',
        'Hammer','Inverted_Hammer','upper_shadow','lower_shadow','upper_shadow_doji','lower_shadow_doji',
        'Outside_Bar','Inside_Bar','Engulf','Bullish_Engulfing','Bearish_Engulfing','Marubozu_Doji','Bullish_Harami','Bearish_Harami',
        'Kangaroo_Tail','Kangaroo_Tail_Bullish','Kangaroo_Tail_Bearish','Partial_Kangaroo_Bullish','Partial_Kangaroo_Bearish',
        'Morning_Star','Evening_Star','Three_White_Soldiers','Three_Black_Crows','Three_White_Soldiers_Doji','Three_Black_Crows_Doji',
        'Kicker','Kanazawa','Kicker_Doji','Kanazawa_Doji','Bullish_Harami_Doji','Bearish_Harami_Doji','Inside_Breakout_Bullish','Inside_Breakout_Bearish'
    ]
    return list(set(pattern_cols))


def get_required_columns():
    indicators = get_indicator_columns()
    sr = get_sr_columns()
    patterns = get_pattern_columns()
    required_columns = list(set(indicators + sr + patterns))
    return required_columns, indicators, sr, patterns


def calculate_indicators(df, pip):
    # Move column assignments into dictionaries and concat at the end
    df = df.sort_values(by='time').reset_index(drop=True)

    df['open'] = df['open'].astype('float32')
    df['high'] = df['high'].astype('float32')
    df['low'] = df['low'].astype('float32')
    df['close'] = df['close'].astype('float32')

    # RSI
    RSIs = [2,7,14,21,50]
    rsi_cols = {}
    for rsi_period in RSIs:
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=rsi_period)
        rsi_cols[f'RSI_{rsi_period}'] = rsi_indicator.rsi().astype('float32')

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    atr_vals = atr_indicator.average_true_range().astype('float32')

    # Bollinger Bands
    #TODO: remove duplicate code (it also exists in calculate_patterns)
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

    bb_cols = {}
    for period, deviation, label in bollinger_settings:
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=period, window_dev=deviation)
        bb_cols[f'{label}_Upper'] = bollinger.bollinger_hband().astype('float32')
        bb_cols[f'{label}_Middle'] = bollinger.bollinger_mavg().astype('float32')
        bb_cols[f'{label}_Lower'] = bollinger.bollinger_lband().astype('float32')
        bb_cols[f'{label}_Bool_Above'] = (df['close'] > bb_cols[f'{label}_Upper'])
        bb_cols[f'{label}_Bool_Below'] = (df['close'] < bb_cols[f'{label}_Lower'])

    # MAs
    moving_averages = {
        'MA_7': 7,
        'MA_21': 21,
        'MA_50': 50,
        'MA_200': 200
    }

    ma_cols = {}
    for ma_label, period in moving_averages.items():
        ma_indicator = ta.trend.SMAIndicator(close=df['close'], window=period)
        ma_cols[ma_label] = ma_indicator.sma_indicator().astype('float32')

    # GA
    df['Is_Green'] = (df['close'] > df['open']).astype('int8')
    ga_cols = {}
    for window, column_name in [(50, 'GA_50'),(100, 'GA_100'),(200, 'GA_200'),(500, 'GA_500')]:
        ga_vals = rolling_sum_numba(df['Is_Green'].values, window) / window
        ga_cols[column_name] = ga_vals

    df.drop(['Is_Green'], axis=1, inplace=True)

    # MA comp
    ma_comp = {}
    for ma_label in ['MA_7', 'MA_21', 'MA_50']:
        ma_comp[f'{ma_label}_comp'] = np.where(
            df['close'] > ma_cols[ma_label], 'above',
            np.where(df['close'] < ma_cols[ma_label], 'below', 'equal')
        )

    # bid/ask
    bid_ask = {}
    pip_val = pip
    bid_ask['bid'] = df['open'] - df['spread'] * pip_val / 2
    bid_ask['ask'] = df['open'] + df['spread'] * pip_val / 2

    # Combine all new indicator columns
    indicator_new_cols = {}
    indicator_new_cols.update(rsi_cols)
    indicator_new_cols['ATR'] = atr_vals
    indicator_new_cols.update(bb_cols)
    indicator_new_cols.update(ma_cols)
    indicator_new_cols.update(ga_cols)
    indicator_new_cols.update(ma_comp)
    indicator_new_cols.update(bid_ask)

    indicator_df = pd.DataFrame(indicator_new_cols, index=df.index)
    df = pd.concat([df, indicator_df], axis=1)

    return df


@njit
def rolling_sum_numba(data, window):
    # Removed parallel=True
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

    period_for_sr = sr_params['period_for_sr']
    touches_for_sr = sr_params['touches_for_sr']
    slack_for_sr_atr_div = sr_params['slack_for_sr_atr_div']
    atr_rejection_multiplier = sr_params['atr_rejection_multiplier']
    min_height_of_sr_distance = sr_params['min_height_of_sr_distance']
    max_height_of_sr_distance = sr_params['max_height_of_sr_distance']

    open_prices = df['open'].values.astype(np.float32)
    high_prices = df['high'].values.astype(np.float32)
    low_prices = df['low'].values.astype(np.float32)
    close_prices = df['close'].values.astype(np.float32)
    atr_values = df['ATR'].values.astype(np.float32)

    upper_sr_array = np.zeros(len(df), dtype=np.float32)
    lower_sr_array = np.zeros(len(df), dtype=np.float32)

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
    n = len(open_prices)
    for i in prange(period_for_sr + 1, n):
        atr = atr_values[i]
        if atr == 0 or np.isnan(atr):
            continue

        uSlackForSR = atr / slack_for_sr_atr_div
        uRejectionFromSR = atr * atr_rejection_multiplier

        current_open = open_prices[i]

        HighSR = current_open + min_height_of_sr_distance * uSlackForSR
        LowSR = current_open - min_height_of_sr_distance * uSlackForSR

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
    counter = 0
    half_rejection = uRejectionFromSR / 2.0

    for idx in range(start_idx, end_idx - 1):
        open_price = open_prices[idx]
        close_price = close_prices[idx]
        high_price = high_prices[idx]
        low_price = low_prices[idx]
        candle_size = abs(high_price - low_price)

        if upper:
            if open_price < current_hline and close_price < current_hline:
                if high_price > current_hline or (
                    candle_size > uRejectionFromSR and (current_hline - high_price) < half_rejection
                ):
                    counter += 1
                    if counter == touches_for_sr:
                        return True
        else:
            if open_price > current_hline and close_price > current_hline:
                if low_price < current_hline or (
                    candle_size > uRejectionFromSR and (low_price - current_hline) < half_rejection
                ):
                    counter += 1
                    if counter == touches_for_sr:
                        return True
    return False



def calculate_all_indicators(df, pip):
    indicator_cols = get_indicator_columns()
    incomplete_mask = df[indicator_cols].isna().any(axis=1)
    if not incomplete_mask.any():
        print("    Indicators: Already complete, skipping.")
        return df

    first_incomplete_idx = np.where(incomplete_mask)[0][0]
    if first_incomplete_idx == 0:
        print("    Indicators: No data calculated before, calculating from start.")
        start_idx = 0
    else:
        start_idx = max(first_incomplete_idx - 510, 0)
        print(f"    Indicators: Partial missing data, starting from index {start_idx}")

    df_to_calc = df.iloc[start_idx:].copy()
    df_to_calc = calculate_indicators(df_to_calc, pip)

    # Ensure that df_to_calc contains all indicator columns
    missing_cols = [col for col in indicator_cols if col not in df_to_calc.columns]
    if missing_cols:
        print(f"    Warning: Missing columns in df_to_calc: {missing_cols}")
        # Optionally, handle missing columns (e.g., fill with default values)
        for col in missing_cols:
            df_to_calc[col] = 0  # or another appropriate default

    # Align indexes so they match the original df's indexing
    df_to_calc.index = df.index[start_idx:start_idx+len(df_to_calc)]

    # Assign columns directly
    for col in indicator_cols:
        df.loc[df_to_calc.index, col] = df_to_calc[col]

    print(f"    Indicators updated from index {start_idx} onwards.")

    return df

def calculate_all_sr_levels(df):
    sr_cols = get_sr_columns()
    incomplete_mask = df[sr_cols].isna().any(axis=1)
    if not incomplete_mask.any():
        print("    SR: Already complete, skipping.")
        return df

    first_incomplete_idx = np.where(incomplete_mask)[0][0]
    if first_incomplete_idx == 0:
        print("    SR: No SR data calculated, calculating from start.")
    else:
        start_idx = max(first_incomplete_idx - 510, 0)
        print(f"    SR: Partial missing data, starting from index {start_idx} (recalculating full)")

    for period_for_sr, touches_for_sr, slack_for_sr_atr_div, atr_rejection_multiplier, config_id in SR_configs:
        upper_col = f"upper_{config_id}"
        lower_col = f"lower_{config_id}"
        sr_incomplete = df[[upper_col, lower_col]].isna().any(axis=1).any()
        if sr_incomplete:
            print(f"    Calculating SR for {config_id}")
            SR_PARAMS = {
                'period_for_sr': period_for_sr,
                'touches_for_sr': touches_for_sr,
                'slack_for_sr_atr_div': slack_for_sr_atr_div,
                'atr_rejection_multiplier': atr_rejection_multiplier
            }
            SR_PARAMS.update(fixed_SR_params)
            df = calculate_sr_levels(df.copy(), SR_PARAMS, upper_col, lower_col)
            df[[upper_col, lower_col]] = df[[upper_col, lower_col]].fillna(0)

    return df


def calculate_all_candle_patterns(df):
    pattern_cols = get_pattern_columns()
    incomplete_mask = df[pattern_cols].isna().any(axis=1)
    if not incomplete_mask.any():
        print("    Patterns: Already complete, skipping.")
        return df

    first_incomplete_idx = np.where(incomplete_mask)[0][0]
    if first_incomplete_idx == 0:
        print("    Patterns: No pattern data calculated before, from start.")
        start_idx = 0
    else:
        start_idx = max(first_incomplete_idx - 510, 0)
        print(f"    Patterns: Partial missing data, start from index {start_idx}")

    df_to_calc = df.iloc[start_idx:].copy()
    df_to_calc = calculate_patterns(df_to_calc)

    # Ensure that df_to_calc contains all pattern columns
    missing_cols = [col for col in pattern_cols if col not in df_to_calc.columns]
    if missing_cols:
        print(f"    Warning: Missing columns in df_to_calc: {missing_cols}")
        # Optionally, handle missing columns (e.g., fill with default values)
        for col in missing_cols:
            df_to_calc[col] = False if 'Bool' in col or col.startswith('Bullish') or col.startswith('Bearish') else 0

    # Align indexes
    df_to_calc.index = df.index[start_idx:start_idx+len(df_to_calc)]


    for col in pattern_cols:
        df.loc[df_to_calc.index, col] = df_to_calc[col]
    print(f"    Patterns updated from index {start_idx} onwards.")
    return df



# For M1 and M5 timeframes: only basic candle patterns
def calculate_basic_patterns(df):
    # Calculate candle_color
    df['candle_color'] = np.where(
        df['close'] > df['open'], 
        1, 
        np.where(df['close'] < df['open'], -1, 0)
    ).astype(np.int8)

    # Calculate necessary boolean columns
    df['Same_Color'] = df['candle_color'] == df['candle_color'].shift(1)
    df['HH'] = df['high'] > df['high'].shift(1)
    df['LL'] = df['low'] < df['low'].shift(1)
    df['HHHC'] = df['HH'] & (df['close'] > df['close'].shift(1))
    df['LLLC'] = df['LL'] & (df['close'] < df['close'].shift(1))

    # Calculate counts
    df['Same_Color_Count'] = count_consecutive(df, 'Same_Color')
    df['HH_Count'] = count_consecutive(df, 'HH')
    df['LL_Count'] = count_consecutive(df, 'LL')
    df['HHHC_Count'] = count_consecutive(df, 'HHHC')
    df['LLLC_Count'] = count_consecutive(df, 'LLLC')

    # Convert pattern columns to bool where applicable
    pattern_bool_cols = [c for c in df.columns if df[c].dtype == bool]
    for c in pattern_bool_cols:
        df[c] = df[c].astype(bool)

    # Convert pattern columns to int8 for all count columns
    pattern_count_cols = [c for c in df.columns if c.endswith('_Count')]
    for c in pattern_count_cols:
        # Fill NaN with 0 before casting
        df[c] = df[c].fillna(0).astype(np.int8)

    return df

def calculate_all_basic_patterns(df):
    # The columns that are required for basic patterns
    basic_pattern_cols = [
        'candle_color','Same_Color','Same_Color_Count','HH','LL','HHHC','LLLC','HH_Count','LL_Count','HHHC_Count','LLLC_Count'
    ]

    # Check if these columns are present and if any are incomplete
    missing_cols = [col for col in basic_pattern_cols if col not in df.columns]
    if missing_cols:
        nan_df = pd.DataFrame({c: np.nan for c in missing_cols}, index=df.index)
        df = pd.concat([df, nan_df], axis=1)

    incomplete_mask = df[basic_pattern_cols].isna().any(axis=1)
    if not incomplete_mask.any():
        print("    Basic Patterns: Already complete, skipping.")
        return df

    first_incomplete_idx = np.where(incomplete_mask)[0][0]
    if first_incomplete_idx == 0:
        print("    Basic Patterns: No data calculated before, from start.")
        start_idx = 0
    else:
        start_idx = max(first_incomplete_idx - 510, 0)
        print(f"    Basic Patterns: Partial missing data, start from index {start_idx}")

    df_to_calc = df.iloc[start_idx:].copy()
    df_to_calc = calculate_basic_patterns(df_to_calc)
    # Ensure that df_to_calc contains all basic pattern columns
    missing_cols = [col for col in basic_pattern_cols if col not in df_to_calc.columns]
    if missing_cols:
        print(f"    Warning: Missing columns in df_to_calc: {missing_cols}")
        # Optionally, handle missing columns (e.g., fill with default values)
        for col in missing_cols:
            df_to_calc[col] = False if 'Bool' in col or col.startswith('Bullish') or col.startswith('Bearish') else 0

    # Align indexes
    df_to_calc.index = df.index[start_idx:start_idx+len(df_to_calc)]

    for col in basic_pattern_cols:
        df.loc[df_to_calc.index, col] = df_to_calc[col]

    print(f"    Basic Patterns updated from index {start_idx} onwards.")

    return df

def process_symbol_timeframe(args):
    symbol, tf_name, output_dir = args
    print(f"Processing symbol: {symbol}, Timeframe: {tf_name}")
    print(f"    Output directory: {output_dir}")
    print(f"time is {datetime.now()}")

    if 'JPY' in symbol:
        pip_digits = 2
    else:
        pip_digits = 4
    pip = 10 ** -pip_digits

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

    required_columns, indicators_cols, sr_cols, pattern_cols = get_required_columns()

    # Ensure all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        nan_df = pd.DataFrame({c: np.nan for c in missing_cols}, index=df.index)
        df = pd.concat([df, nan_df], axis=1)
        df = df.copy()

    # 1. Indicators
    if tf_name in ['M15', 'M30', 'H1', 'H4', 'D1', 'W1']:
        print(f"    Calculating indicators for {symbol} {tf_name}, time is {datetime.now()}")
        df = calculate_all_indicators(df, pip)

    # 2. SR
    if tf_name in ['M15', 'M30', 'H1', 'H4', 'D1', 'W1']:
        print(f"    Calculating SR levels for {symbol} {tf_name}, time is {datetime.now()}")
        df = calculate_all_sr_levels(df)

    # 3. Patterns - choose basic or full pattern calculation
    print(f"    Calculating patterns for {symbol} {tf_name}, time is {datetime.now()}")
    if tf_name in ['M1', 'M5']:
        df = calculate_all_basic_patterns(df)
    else:
        df = calculate_all_candle_patterns(df)

    try:
        df.to_parquet(filepath, index=False)
        print(f"    Updated data saved to {filepath}")
    except Exception as e:
        print(f"    Error saving data to Parquet: {e}")

def calculate_indicators_for_files():
    output_dir = os.path.join(drive, folder)
    if not os.path.exists(output_dir):
        print(f"Data directory {output_dir} does not exist.")
        return
    
    for symbol in currency_pairs:
        for tf_name in timeframes:
            process_symbol_timeframe((symbol, tf_name, output_dir))

    #multiprocessing option
    """ 
    tasks = []
    for symbol in currency_pairs:
        for tf_name in timeframes:
            tasks.append((symbol, tf_name, output_dir))

    num_processes = 2
    with Pool(processes=num_processes) as pool:
        pool.map(process_symbol_timeframe, tasks)
    """



if __name__ == "__main__":
    calculate_indicators_for_files()