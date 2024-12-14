# calculate_history_indicators.py

"""
This script reads historical data files for currency pairs and timeframes,
calculates technical indicators, SR levels, and candle patterns, and updates the data files with new columns.
It utilizes Numba for JIT compilation and multiprocessing for parallel processing to optimize performance.

Logic:
- Split into 3 groups: Indicators, SR, Candle Patterns.
- If no data for a column in the group: calculate from start.
- If partial data missing: find first incomplete index and recalculate only from that index - 510.
- If full data: no recalculation.

Refactored:  
All functions now directly modify 'df' in place.  
No function returns 'df', and no intermediate DataFrame copies for assignment.
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

from .calculate_history_indicators import (check_bb_return_long, check_bb_return_short, check_bb_over_long, check_bb_over_short,
                                             check_ma_crossover_long, check_ma_crossover_short, check_price_ma_crossover_long,   check_price_ma_crossover_short,
                                             check_sr_long, check_sr_short, check_breakout_long, check_breakout_short, check_fakeout_long, check_fakeout_short,
                                             check_bars_trend_long, check_bars_trend_short)

# Recalculation Flags (Set to True to force full recalculation)
FORCE_RECALC_INDICATORS = False
FORCE_RECALC_SR = False
FORCE_RECALC_PATTERNS = False
FORCE_RECALC_INDICATORS_DECISIONS = False
FORCE_RECALC_TARGETS = False

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

    counts[0] = 1
    current_count = 1

    for i in range(1, n):
        condition = False

        if comparison_type == MARUBOZU:
            if df.at[i, 'Marubozu'] and df.at[i, 'candle_color'] == df.at[i - 1, 'candle_color']:
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


def get_indicator_columns():
    indicator_cols = ['open', 'high', 'low', 'close', 'spread', 'time', 'bid', 'ask', 'ATR']
    RSIs = [2, 7, 14, 21, 50]
    for r in RSIs:
        indicator_cols.append(f'RSI_{r}')
    bb_settings = [
        (15, 1.5, 'BB15_1.5'), (15, 2.0, 'BB15_2.0'), (15, 2.5, 'BB15_2.5'),
        (20, 1.5, 'BB20_1.5'), (20, 2.0, 'BB20_2.0'), (20, 2.5, 'BB20_2.5'),
        (25, 1.5, 'BB25_1.5'), (25, 2.0, 'BB25_2.0'), (25, 2.5, 'BB25_2.5'),
    ]
    for period, deviation, label in bb_settings:
        indicator_cols.extend([f'{label}_Upper', f'{label}_Middle', f'{label}_Lower', f'{label}_Bool_Above', f'{label}_Bool_Below'])

    for p in [7, 21, 50, 200]:
        indicator_cols.append(f'MA_{p}')
        if p in [7, 21, 50]:
            indicator_cols.append(f'MA_{p}_comp')
    for w in [50, 100, 200, 500]:
        indicator_cols.append(f'GA_{w}')

    # Ensure uniqueness and order
    indicator_cols = list(dict.fromkeys(indicator_cols))
    return indicator_cols


def get_sr_columns():
    sr_cols = []
    period = [75, 200, 500]
    touches = [3, 4, 5]
    slack_div = [5, 10, 15]
    rejection_multi = [0.5, 1.0, 1.5]

    # SR_configs inline:
    for r in rejection_multi:
        for t in slack_div:
            for s in touches:
                for l in period:
                    config_id = f"SR{l}_{s}_{t}_{r}"
                    sr_cols.append(f'upper_{config_id}')
                    sr_cols.append(f'lower_{config_id}')

    sr_cols = list(dict.fromkeys(sr_cols))
    return sr_cols


def get_pattern_columns():
    pattern_cols = [
        'candle_color', 'Body_Size', 'Candle_Size', 'Upper_Wik_Size', 'Lower_Wik_Size', 'Upper_wik_ratio', 'Lower_wik_ratio', 'wik_ratio',
        'Doji', 'Marubozu', 'Bullish_Marubozu', 'Bearish_Marubozu', 'Marubozu_Count', 'Bullish_Marubozu_Count', 'Bearish_Marubozu_Count',
        'Same_Color', 'Same_Color_Count', 'HH', 'LL', 'HHHC', 'LLLC', 'HH_Count', 'LL_Count', 'HHHC_Count', 'LLLC_Count',
        'Hammer', 'Inverted_Hammer', 'upper_shadow', 'lower_shadow', 'upper_shadow_doji', 'lower_shadow_doji',
        'Outside_Bar', 'Inside_Bar', 'Engulf', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Marubozu_Doji', 'Bullish_Harami', 'Bearish_Harami',
        'Kangaroo_Tail', 'Kangaroo_Tail_Bullish', 'Kangaroo_Tail_Bearish', 'Partial_Kangaroo_Bullish', 'Partial_Kangaroo_Bearish',
        'Morning_Star', 'Evening_Star', 'Three_White_Soldiers', 'Three_Black_Crows', 'Three_White_Soldiers_Doji', 'Three_Black_Crows_Doji',
        'Kicker', 'Kanazawa', 'Kicker_Doji', 'Kanazawa_Doji', 'Bullish_Harami_Doji', 'Bearish_Harami_Doji', 'Inside_Breakout_Bullish', 'Inside_Breakout_Bearish'
    ]
    pattern_cols = list(dict.fromkeys(pattern_cols))
    return pattern_cols

def get_inidicator_decision_columns():
    #bb_decisions 
    indicator_cols = []
    periods= [15, 20, 25]
    deviations= [1.5, 2.0, 2.5]
    strategies = ['with_long', 'with_short', 'return_long', 'return_short', 'over_long', 'over_short']
    for period in periods:
        for deviation in deviations:
            for strategy in strategies:
                indicator_cols.append(f'BB{period}_{deviation}_{strategy}')

    #ma_decisions
    settings = ['7vs21vs50' , '21vs50vs200']
    for setting in settings:
        indicator_cols.append(f'MA_Cross_with_long_{setting}')
        indicator_cols.append(f'MA_Cross_with_short_{setting}')
    MAs = [7, 21, 50, 200]
    for ma in MAs:
        indicator_cols.append(f'MA_Cross_price_with_long_{ma}')
        indicator_cols.append(f'MA_Cross_price_with_short_{ma}')
    
    # Range Decisions
    period = [75, 200, 500]
    touches = [3, 4, 5]
    slack_div = [5, 10, 15]
    rejection_multi = [0.5, 1.0, 1.5]
    for r in rejection_multi:
        for t in slack_div:
            for s in touches:
                for l in period:
                    #     strategies = ['SR', 'Breakout', 'Fakeout'] 
                    config_id = f"SR{l}_{s}_{t}_{r}"
                    indicator_cols.append(f'SR_long_{config_id}')
                    indicator_cols.append(f'SR_short_{config_id}')
                    indicator_cols.append(f'Breakout_long_{config_id}')
                    indicator_cols.append(f'Breakout_short_{config_id}')
                    indicator_cols.append(f'Fakeout_long_{config_id}')
                    indicator_cols.append(f'Fakeout_short_{config_id}')
    
    # Bars Trend
    periods = [20, 50, 100]
    for period in periods:
        indicator_cols.append(f'Bars_Trend_long_{period}')
        indicator_cols.append(f'Bars_Trend_short_{period}')


def get_traget_columns():
    pass
    #TODO: Add target columns for indicators

def get_required_columns():
    indicators = get_indicator_columns()
    sr = get_sr_columns()
    patterns = get_pattern_columns()
    indicators_decisions = get_inidicator_decision_columns()
    targets = get_traget_columns()
    required_columns = list(dict.fromkeys(indicators + sr + patterns + indicators_decisions + targets))
    return required_columns


@njit
def rolling_sum_numba(data, window):
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


def calculate_indicators(df, pip, start_idx=0):
    # Operate only on df from start_idx onwards
    close_sub = df['close'].iloc[start_idx:]
    high_sub = df['high'].iloc[start_idx:]
    low_sub = df['low'].iloc[start_idx:]
    open_sub = df['open'].iloc[start_idx:]
    spread_sub = df['spread'].iloc[start_idx:]

    # RSI
    RSIs = [2, 7, 14, 21, 50]
    for rsi_period in RSIs:
        rsi_indicator = ta.momentum.RSIIndicator(close=close_sub, window=rsi_period)
        df.loc[start_idx:, f'RSI_{rsi_period}'] = rsi_indicator.rsi().values.astype('float32')

    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(high=high_sub, low=low_sub, close=close_sub, window=14)
    df.loc[start_idx:, 'ATR'] = atr_indicator.average_true_range().values.astype('float32')

    # Bollinger Bands
    bb_settings = [
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

    for period, deviation, label in bb_settings:
        bollinger = ta.volatility.BollingerBands(close=close_sub, window=period, window_dev=deviation)
        upper = bollinger.bollinger_hband().values.astype('float32')
        mid = bollinger.bollinger_mavg().values.astype('float32')
        lower = bollinger.bollinger_lband().values.astype('float32')
        df.loc[start_idx:, f'{label}_Upper'] = upper
        df.loc[start_idx:, f'{label}_Middle'] = mid
        df.loc[start_idx:, f'{label}_Lower'] = lower
        # Bool_Above and Bool_Below
        # TODO: Check if this can be optimized (int8 or bool)
        df.loc[start_idx:, f'{label}_Bool_Above'] = (df['close'].iloc[start_idx:].values > upper)
        df.loc[start_idx:, f'{label}_Bool_Below'] = (df['close'].iloc[start_idx:].values < lower)

    # MAs
    moving_averages = {
        'MA_7': 7,
        'MA_21': 21,
        'MA_50': 50,
        'MA_200': 200
    }

    ma_results = {}
    for ma_label, period in moving_averages.items():
        ma_indicator = ta.trend.SMAIndicator(close=close_sub, window=period)
        ma_data = ma_indicator.sma_indicator().values.astype('float32')
        df.loc[start_idx:, ma_label] = ma_data
        ma_results[ma_label] = ma_data

    # GA
    is_green = (df['close'].iloc[start_idx:].values > df['open'].iloc[start_idx:].values).astype('int8')
    for window, column_name in [(50, 'GA_50'), (100, 'GA_100'), (200, 'GA_200'), (500, 'GA_500')]:
        ga_vals = rolling_sum_numba(is_green, window) / window
        df.loc[start_idx:, column_name] = ga_vals

    # MA comp
    for ma_label in ['MA_7', 'MA_21', 'MA_50']:
        close_vals = df['close'].iloc[start_idx:].values
        ma_vals = df[ma_label].iloc[start_idx:].values
        comp = np.where(close_vals > ma_vals, 'above', np.where(close_vals < ma_vals, 'below', 'equal'))
        df.loc[start_idx:, f'{ma_label}_comp'] = comp

    # bid/ask
    pip_val = pip
    open_vals = df['open'].iloc[start_idx:].values
    spread_vals = spread_sub.values
    df.loc[start_idx:, 'bid'] = open_vals - spread_vals * pip_val / 2
    df.loc[start_idx:, 'ask'] = open_vals + spread_vals * pip_val / 2


def calculate_all_indicators(df, pip):
    global FORCE_RECALC_INDICATORS  # Access the global flag
    indicator_cols = get_indicator_columns()

    if FORCE_RECALC_INDICATORS:
        print("    Indicators: Full recalculation requested. Calculating from start.")
        start_idx = 0
    else:
        incomplete_mask = df[indicator_cols].isna().any(axis=1)
        if not incomplete_mask.any():
            print("    Indicators: Already complete, skipping.")
            return

        first_incomplete_idx = np.where(incomplete_mask)[0][0]
        if first_incomplete_idx == 0:
            print("    Indicators: No data calculated before, calculating from start.")
            start_idx = 0
        else:
            start_idx = max(first_incomplete_idx - 510, 0)
            print(f"    Indicators: Partial missing data, starting from index {start_idx}")

    calculate_indicators(df, pip, start_idx=start_idx)
    print(f"    Indicators updated from index {start_idx} onwards.")


def calculate_patterns(df, start_idx=0):
    # entire df might be used, but focus from start_idx
    doji_ratio = 0.1
    upper_quarter_threshold = 0.75
    lower_quarter_threshold = 0.25
    wik_ratio_threshold = 0.25
    marubozu_threshold = 2.5

    # candle_color
    df.loc[start_idx:, 'candle_color'] = np.where(
        df['close'].iloc[start_idx:] > df['open'].iloc[start_idx:], 1,
        np.where(df['close'].iloc[start_idx:] < df['open'].iloc[start_idx:], -1, 0)
    ).astype(np.int8)

    # compute candle measures
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    close_arr = df['close'].values
    color_arr = df['candle_color'].values

    body_size, candle_size, upper_wik_size, lower_wik_size, upper_wik_ratio, lower_wik_ratio, wik_ratio = compute_candle_measures(
        open_arr, high_arr, low_arr, close_arr, color_arr
    )

    df.loc[start_idx:, 'Body_Size'] = body_size[start_idx:]
    df.loc[start_idx:, 'Candle_Size'] = candle_size[start_idx:]
    df.loc[start_idx:, 'Upper_Wik_Size'] = upper_wik_size[start_idx:]
    df.loc[start_idx:, 'Lower_Wik_Size'] = lower_wik_size[start_idx:]
    df.loc[start_idx:, 'Upper_wik_ratio'] = upper_wik_ratio[start_idx:]
    df.loc[start_idx:, 'Lower_wik_ratio'] = lower_wik_ratio[start_idx:]
    df.loc[start_idx:, 'wik_ratio'] = wik_ratio[start_idx:]

    # patterns
    df.loc[start_idx:, 'HH'] = (df['high'].shift(1) < df['high'])[start_idx:]
    df.loc[start_idx:, 'LL'] = (df['low'].shift(1) > df['low'])[start_idx:]
    df.loc[start_idx:, 'HHHC'] = (df['HH'] & (df['close'] > df['close'].shift(1)))[start_idx:]
    df.loc[start_idx:, 'LLLC'] = (df['LL'] & (df['close'] < df['close'].shift(1)))[start_idx:]
    df.loc[start_idx:, 'Same_Color'] = (df['candle_color'] == df['candle_color'].shift(1))[start_idx:]

    body_center = (df['open'] + df['close']) / 2
    relative_pos = (body_center - df['low']) / df['Candle_Size']
    df.loc[start_idx:, 'Doji'] = ((df['Body_Size'] < df['Candle_Size'] * doji_ratio) &
                                  (relative_pos >= lower_quarter_threshold) &
                                  (relative_pos <= upper_quarter_threshold))[start_idx:]

    df.loc[start_idx:, 'Marubozu'] = ((np.maximum(df['Upper_wik_ratio'], df['Lower_wik_ratio']) >= marubozu_threshold) &
                                      (df['wik_ratio'] > 1.75) &
                                      (df['candle_color'] != 0))[start_idx:]
    df.loc[start_idx:, 'Bullish_Marubozu'] = (df['Marubozu'] & (df['candle_color'] == 1))[start_idx:]
    df.loc[start_idx:, 'Bearish_Marubozu'] = (df['Marubozu'] & (df['candle_color'] == -1))[start_idx:]

    df.loc[start_idx:, 'Hammer'] = ((df['wik_ratio'] <= wik_ratio_threshold) &
                                    ((df['Upper_Wik_Size'] == 0) | ((df['Lower_Wik_Size'] / df['Upper_Wik_Size']) > 2)))[start_idx:]
    df.loc[start_idx:, 'Inverted_Hammer'] = ((df['wik_ratio'] <= wik_ratio_threshold) &
                                             ((df['Lower_Wik_Size'] == 0) | ((df['Upper_Wik_Size'] / df['Lower_Wik_Size']) > 2)))[start_idx:]

    df.loc[start_idx:, 'upper_shadow'] = (df['Upper_Wik_Size'] > df['Body_Size'] * wik_ratio_threshold)[start_idx:]
    df.loc[start_idx:, 'lower_shadow'] = (df['Lower_Wik_Size'] > df['Body_Size'] * wik_ratio_threshold)[start_idx:]
    df.loc[start_idx:, 'upper_shadow_doji'] = (df['upper_shadow'] & df['Doji'])[start_idx:]
    df.loc[start_idx:, 'lower_shadow_doji'] = (df['lower_shadow'] & df['Doji'])[start_idx:]

    df.loc[start_idx:, 'Outside_Bar'] = (df['HH'].shift(1) & df['LL'].shift(1))[start_idx:]
    df.loc[start_idx:, 'Inside_Bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1)))[start_idx:]

    df.loc[start_idx:, 'Engulf'] = ((df['candle_color'] != df['candle_color'].shift(1)) &
                                    (((df['candle_color'] == 1) &
                                      (df['open'] <= df['close'].shift(1)) &
                                      (df['close'] >= df['open'].shift(1))) |
                                     ((df['candle_color'] == -1) &
                                      (df['open'] >= df['close'].shift(1)) &
                                      (df['close'] <= df['open'].shift(1)))))[start_idx:]
    df.loc[start_idx:, 'Bullish_Engulfing'] = ((df['candle_color'].shift(1) == -1) &
                                               (df['candle_color'] == 1) & df['Engulf'])[start_idx:]
    df.loc[start_idx:, 'Bearish_Engulfing'] = ((df['candle_color'].shift(1) == 1) &
                                               (df['candle_color'] == -1) & df['Engulf'])[start_idx:]
    df.loc[start_idx:, 'Marubozu_Doji'] = (df['Marubozu'] & df['Doji'].shift(-1))[start_idx:]

    df.loc[start_idx:, 'Bullish_Harami'] = ((df['candle_color'].shift(1) == -1) &
                                            (df['candle_color'] == 1) & df['Inside_Bar'])[start_idx:]
    df.loc[start_idx:, 'Bearish_Harami'] = ((df['candle_color'].shift(1) == 1) &
                                            (df['candle_color'] == -1) & df['Inside_Bar'])[start_idx:]

    df.loc[start_idx:, 'HHHL'] = ((df['high'] > df['high'].shift(1)) &
                                  (df['low'] > df['low'].shift(1)))[start_idx:]
    df.loc[start_idx:, 'LHLL'] = ((df['high'] < df['high'].shift(1)) &
                                  (df['low'] < df['low'].shift(1)))[start_idx:]

    df.loc[start_idx:, 'Kangaroo_Tail'] = (((df['candle_color'].shift(1) == 1) & df['LHLL'].shift(2) & df['HHHL'].shift(1)) |
                                           ((df['candle_color'].shift(1) == -1) & df['HHHL'].shift(2) & df['LHLL'].shift(1)))[start_idx:]
    df.loc[start_idx:, 'Kangaroo_Tail_Bullish'] = (df['Kangaroo_Tail'] & (df['candle_color'] == 1))[start_idx:]
    df.loc[start_idx:, 'Kangaroo_Tail_Bearish'] = (df['Kangaroo_Tail'] & (df['candle_color'] == -1))[start_idx:]

    df.loc[start_idx:, 'Partial_Kangaroo_Bullish'] = ((df['low'].shift(2) < df['low'].shift(3)) &
                                                      (df['low'].shift(2) < df['low'].shift(1)))[start_idx:]
    df.loc[start_idx:, 'Partial_Kangaroo_Bearish'] = ((df['high'].shift(2) > df['high'].shift(3)) &
                                                      (df['high'].shift(2) > df['high'].shift(1)))[start_idx:]

    df.loc[start_idx:, 'Morning_Star'] = ((df['candle_color'].shift(2) == -1) & (df['Doji'].shift(1)) &
                                          (df['candle_color'] == 1))[start_idx:]
    df.loc[start_idx:, 'Evening_Star'] = ((df['candle_color'].shift(2) == 1) & (df['Doji'].shift(1)) &
                                          (df['candle_color'] == -1))[start_idx:]

    df.loc[start_idx:, 'Three_White_Soldiers'] = ((df['candle_color'].shift(2) == 1) &
                                                  (df['candle_color'].shift(1) == 1) &
                                                  (df['candle_color'] == 1))[start_idx:]
    df.loc[start_idx:, 'Three_Black_Crows'] = ((df['candle_color'].shift(2) == -1) &
                                               (df['candle_color'].shift(1) == -1) &
                                               (df['candle_color'] == -1))[start_idx:]
    df.loc[start_idx:, 'Three_White_Soldiers_Doji'] = (df['Three_White_Soldiers'] & df['Doji'].shift(-1))[start_idx:]
    df.loc[start_idx:, 'Three_Black_Crows_Doji'] = (df['Three_Black_Crows'] & df['Doji'].shift(-1))[start_idx:]

    df.loc[start_idx:, 'Kicker'] = ((df['candle_color'].shift(1) == 1) & (df['candle_color'] == -1))[start_idx:]
    df.loc[start_idx:, 'Kanazawa'] = ((df['candle_color'].shift(1) == -1) & (df['candle_color'] == 1))[start_idx:]
    df.loc[start_idx:, 'Kicker_Doji'] = (df['Kicker'] & df['Doji'].shift(-1))[start_idx:]
    df.loc[start_idx:, 'Kanazawa_Doji'] = (df['Kanazawa'] & df['Doji'].shift(-1))[start_idx:]

    df.loc[start_idx:, 'Bullish_Harami_Doji'] = (df['Bearish_Harami'] & df['Doji'].shift(-1))[start_idx:]
    df.loc[start_idx:, 'Bearish_Harami_Doji'] = (df['Bullish_Harami'] & df['Doji'].shift(-1))[start_idx:]

    df.loc[start_idx:, 'Inside_Breakout_Bullish'] = ((df['Inside_Bar'].shift(2)) &
                                                     (df['close'].shift(1) > df['high'].shift(3)))[start_idx:]
    df.loc[start_idx:, 'Inside_Breakout_Bearish'] = ((df['Inside_Bar'].shift(2)) &
                                                     (df['close'].shift(1) < df['low'].shift(3)))[start_idx:]

    # counts
    # Use full df because count_consecutive looks at full series
    df['Same_Color_Count'] = count_consecutive(df, SAME_COLOR)
    df['Marubozu_Count'] = count_consecutive(df, MARUBOZU)
    df['Bullish_Marubozu_Count'] = count_consecutive(df, BULLISH_MARUBOZU)
    df['Bearish_Marubozu_Count'] = count_consecutive(df, BEARISH_MARUBOZU)
    df['HH_Count'] = count_consecutive(df, HH)
    df['LL_Count'] = count_consecutive(df, LL)
    df['HHHC_Count'] = count_consecutive(df, HHHC)
    df['LLLC_Count'] = count_consecutive(df, LLLC)

    pattern_bool_cols = [c for c in df.columns if df[c].dtype == bool or df[c].dtype == np.bool_]
    for c in pattern_bool_cols:
        df[c] = df[c].astype(bool)

    pattern_count_cols = [c for c in df.columns if c.endswith('_Count')]
    for c in pattern_count_cols:
        df[c] = df[c].fillna(0).astype(np.int8)


def calculate_all_candle_patterns(df):
    global FORCE_RECALC_PATTERNS  # Access the global flag
    pattern_cols = get_pattern_columns()

    if FORCE_RECALC_PATTERNS:
        print("    Candle Patterns: Full recalculation requested. Calculating from start.")
        start_idx = 0
    else:
        incomplete_mask = df[pattern_cols].isna().any(axis=1)
        if not incomplete_mask.any():
            print("    Patterns: Already complete, skipping.")
            return

        first_incomplete_idx = np.where(incomplete_mask)[0][0]
        if first_incomplete_idx == 0:
            print("    Patterns: No pattern data calculated before, from start.")
            start_idx = 0
        else:
            start_idx = max(first_incomplete_idx - 510, 0)
            print(f"    Patterns: Partial missing data, start from index {start_idx}")

    calculate_patterns(df, start_idx=start_idx)
    print(f"    Patterns updated from index {start_idx} onwards.")


def calculate_basic_patterns(df, start_idx=0):
    df.loc[start_idx:, 'candle_color'] = np.where(
        df['close'].iloc[start_idx:] > df['open'].iloc[start_idx:], 1,
        np.where(df['close'].iloc[start_idx:] < df['open'].iloc[start_idx:], -1, 0)
    ).astype(np.int8)

    df.loc[start_idx:, 'Same_Color'] = (df['candle_color'] == df['candle_color'].shift(1))[start_idx:]
    df.loc[start_idx:, 'HH'] = (df['high'] > df['high'].shift(1))[start_idx:]
    df.loc[start_idx:, 'LL'] = (df['low'] < df['low'].shift(1))[start_idx:]
    df.loc[start_idx:, 'HHHC'] = (df['HH'] & (df['close'] > df['close'].shift(1)))[start_idx:]
    df.loc[start_idx:, 'LLLC'] = (df['LL'] & (df['close'] < df['close'].shift(1)))[start_idx:]

    df['Same_Color_Count'] = count_consecutive(df, SAME_COLOR)
    df['HH_Count'] = count_consecutive(df, HH)
    df['LL_Count'] = count_consecutive(df, LL)
    df['HHHC_Count'] = count_consecutive(df, HHHC)
    df['LLLC_Count'] = count_consecutive(df, LLLC)

    pattern_bool_cols = [c for c in df.columns if df[c].dtype == bool]
    for c in pattern_bool_cols:
        df[c] = df[c].astype(bool)

    pattern_count_cols = [c for c in df.columns if c.endswith('_Count')]
    for c in pattern_count_cols:
        df[c] = df[c].fillna(0).astype(np.int8)


def calculate_all_basic_patterns(df):
    global FORCE_RECALC_PATTERNS  # Access the global flag
    basic_pattern_cols = [
        'candle_color', 'Same_Color', 'Same_Color_Count', 'HH', 'LL', 'HHHC', 'LLLC', 'HH_Count', 'LL_Count', 'HHHC_Count', 'LLLC_Count'
    ]

    missing_cols = [col for col in basic_pattern_cols if col not in df.columns]
    if missing_cols:
        nan_df = pd.DataFrame({c: np.nan for c in missing_cols}, index=df.index)
        df = pd.concat([df, nan_df], axis=1)

    if FORCE_RECALC_PATTERNS:
        print("    Basic Patterns: Full recalculation requested. Calculating from start.")
        start_idx = 0
    else:
        incomplete_mask = df[basic_pattern_cols].isna().any(axis=1)
        if not incomplete_mask.any():
            print("    Basic Patterns: Already complete, skipping.")
            return

        first_incomplete_idx = np.where(incomplete_mask)[0][0]
        if first_incomplete_idx == 0:
            print("    Basic Patterns: No data calculated before, from start.")
            start_idx = 0
        else:
            start_idx = max(first_incomplete_idx - 510, 0)
            print(f"    Basic Patterns: Partial missing data, start from index {start_idx}")

    calculate_basic_patterns(df, start_idx=start_idx)
    print(f"    Basic Patterns updated from index {start_idx} onwards.")


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

        # Upper SR
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

        # Lower SR
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
                if high_price > current_hline or (candle_size > uRejectionFromSR and (current_hline - high_price) < half_rejection):
                    counter += 1
                    if counter == touches_for_sr:
                        return True
        else:
            if open_price > current_hline and close_price > current_hline:
                if low_price < current_hline or (candle_size > uRejectionFromSR and (low_price - current_hline) < half_rejection):
                    counter += 1
                    if counter == touches_for_sr:
                        return True
    return False


def calculate_sr_levels(df, sr_params, upper_sr_col, lower_sr_col):
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


def calculate_all_sr_levels(df):
    global FORCE_RECALC_SR  # Access the global flag
    sr_cols = get_sr_columns()

    if FORCE_RECALC_SR:
        print("    SR Levels: Full recalculation requested. Calculating from start.")
        # Reset SR columns to NaN to force recalculation
        df[sr_cols] = np.nan
        start_idx = 0
    else:
        incomplete_mask = df[sr_cols].isna().any(axis=1)
        if not incomplete_mask.any():
            print("    SR: Already complete, skipping.")
            return

        first_incomplete_idx = np.where(incomplete_mask)[0][0]
        if first_incomplete_idx == 0:
            print("    SR: No SR data calculated, calculating from start.")
            start_idx = 0
        else:
            start_idx = max(first_incomplete_idx - 510, 0)
            print(f"    SR: Partial missing data, starting from index {start_idx} (recalculating full)")

    # We always recalculate SR PARAMS from start for simplicity
    period = [75, 200, 500]
    touches = [3, 4, 5]
    slack_div = [5, 10, 15]
    rejection_multi = [0.5, 1.0, 1.5]
    fixed_SR_params = {
        'min_height_of_sr_distance': 3.0,
        'max_height_of_sr_distance': 60.0,
    }

    for r in rejection_multi:
        for t in slack_div:
            for s in touches:
                for l in period:
                    config_id = f"SR{l}_{s}_{t}_{r}"
                    upper_col = f"upper_{config_id}"
                    lower_col = f"lower_{config_id}"
                    sr_incomplete = df[[upper_col, lower_col]].isna().any(axis=1).any()
                    if sr_incomplete or FORCE_RECALC_SR:
                        print(f"    Calculating SR for {config_id}")
                        SR_PARAMS = {
                            'period_for_sr': l,
                            'touches_for_sr': s,
                            'slack_for_sr_atr_div': t,
                            'atr_rejection_multiplier': r
                        }
                        SR_PARAMS.update(fixed_SR_params)
                        calculate_sr_levels(df, SR_PARAMS, upper_col, lower_col)
                        df[[upper_col, lower_col]] = df[[upper_col, lower_col]].fillna(0)

def get_relevant_columns_by_timeframe(tf_name):
    # Get all sets of columns
    required_columns, indicators_cols, sr_cols, pattern_cols = get_required_columns()

    # For M1 and M5, we only need basic pattern columns
    basic_pattern_cols = [
        'candle_color','Same_Color','Same_Color_Count','HH','LL','HHHC','LLLC','HH_Count','LL_Count','HHHC_Count','LLLC_Count'
    ]
    
    if tf_name in ['M1', 'M5']:
        relevant_cols = basic_pattern_cols
    else:
        # For M15 and above, all are relevant: indicators, SR, and full patterns
        relevant_cols = required_columns
    
    # Ensure uniqueness
    relevant_cols = list(dict.fromkeys(relevant_cols))
    return relevant_cols

def calculate_all_indicator_decisions(df, pip, start_idx=0):
    # TODO: Check if this can be optimized (int8 or bool)
    # TODO: Implement this function
    
    # Bollinger Bands
    bb_settings = [
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

    for period, deviation, label in bb_settings:
        # bb with strategy - buy if previous candle closed above upper band, sell if closed below lower band
        # make use of the precalculated {label}_Bool_Above , {label}_Bool_Below columns
        # since logic is very simple forgo the use of check_bb_with_long and check_bb_with_short functions
        df.loc[start_idx:, f'{label}_with_long'] = df[f'{label}_Bool_Above'].shift(1).astype('int8')
        df.loc[start_idx:, f'{label}_with_short'] = df[f'{label}_Bool_Below'].shift(1).astype('int8')

        # bb return_long and return_short
        # use the check_bb_return_long , and check_bb_return_short functions
        # parameters: previous bool below/above, previous previous bool below/above - i.e. last and last last bool below/above
        df.loc[start_idx:, f'{label}_return_long'] = check_bb_return_long(df[f'{label}_Bool_Below'].shift(1).values, df[f'{label}_Bool_Below'].shift(2).values).astype('int8') 
        df.loc[start_idx:, f'{label}_return_short'] = check_bb_return_short(df[f'{label}_Bool_Above'].shift(1).values, df[f'{label}_Bool_Above'].shift(2).values).astype('int8')

        # bb over long and short
        # use the check_bb_over_long , and check_bb_over_long functions
        # parameters: prev close, prev prev close, prev middle band, prev prev middle band
        df.loc[start_idx:, f'{label}_over_long'] = check_bb_over_long(df['close'].shift(1).values, df['close'].shift(2).values, df[f'{label}_Middle'].shift(1).values, df[f'{label}_Middle'].shift(2).values).astype('int8')
        df.loc[start_idx:, f'{label}_over_short'] = check_bb_over_short(df['close'].shift(1).values, df['close'].shift(2).values, df[f'{label}_Middle'].shift(1).values, df[f'{label}_Middle'].shift(2).values).astype('int8')

    # MA cross

    # MA crossover strategy, use the check_ma_crossover_long, and check_ma_crossover_short functions
    # parameters: prev_short_ma, prev_prev_short_ma, prev_medium_ma,prev_prev_medium_ma, prev_long_ma 
    # for long:     Buy if short-term MA crosses above medium-term MA and medium-term MA is below long-term MA. - buy into strength after a pullback
    # for short:    Sell if short-term MA crosses below medium-term MA and medium-term MA is above long-term MA. - sell into weakness after a bounce
    settings = ['7vs21vs50' , '21vs50vs200']
    for setting in settings:
        short_ma, medium_ma, long_ma = setting.split('vs')
        short_ma = int(short_ma)
        medium_ma = int(medium_ma)
        long_ma = int(long_ma)
        df.loc[start_idx:, f'MA_Cross_with_long_{setting}'] = check_ma_crossover_long(df[f'MA_{short_ma}'].shift(1).values, df[f'MA_{short_ma}'].shift(2).values, df[f'MA_{medium_ma}'].shift(1).values, df[f'MA_{medium_ma}'].shift(2).values, df[f'MA_{long_ma}'].shift(1).values).astype('int8')
        df.loc[start_idx:, f'MA_Cross_with_short_{setting}'] = check_ma_crossover_short(df[f'MA_{short_ma}'].shift(1).values, df[f'MA_{short_ma}'].shift(2).values, df[f'MA_{medium_ma}'].shift(1).values, df[f'MA_{medium_ma}'].shift(2).values, df[f'MA_{long_ma}'].shift(1).values).astype('int8')

    # MA price cross
    # parameters: prev close, prev prev close, prev ma, prev ma
    MAs = [7, 21, 50, 200]
    for ma in MAs:
        df.loc[start_idx:, f'MA_Cross_price_with_long_{ma}'] = check_price_ma_crossover_long(df['close'].shift(1).values, df['close'].shift(2).values, df[f'MA_{ma}'].shift(1).values, df[f'MA_{ma}'].shift(2).values).astype('int8')
        df.loc[start_idx:, f'MA_Cross_price_with_short_{ma}'] = check_price_ma_crossover_short(df['close'].shift(1).values, df['close'].shift(2).values, df[f'MA_{ma}'].shift(1).values, df[f'MA_{ma}'].shift(2).values).astype('int8')


    # Range
    period = [75, 200, 500]
    touches = [3, 4, 5]
    slack_div = [5, 10, 15]
    rejection_multi = [0.5, 1.0, 1.5]
    strategies = ['SR', 'Breakout', 'Fakeout'] 
    for r in rejection_multi:
        for s in slack_div:
            for t in touches:
                for p in period:
                    config_id = f"SR{p}_{t}_{s}_{r}"
                    upper_col = f"upper_{config_id}"
                    lower_col = f"lower_{config_id}"

                    strategy = 'SR'
                    # check_sr_long(prev_lower_sr, prev_close, prev_ATR):
                    df.loc[start_idx:, f'{strategy}_long_{config_id}'] = check_sr_long(df[lower_col].shift(1).values, df['close'].shift(1).values, df['ATR'].shift(1).values).astype('int8')
                    df.loc[start_idx:, f'{strategy}_short_{config_id}'] = check_sr_short(df[upper_col].shift(1).values, df['close'].shift(1).values, df['ATR'].shift(1).values).astype('int8')
                    
                    strategy = 'Breakout'
                    # check_breakout_long(prev_upper_sr, prev_close, prev_ATR):
                    df.loc[start_idx:, f'{strategy}_long_{config_id}'] = check_breakout_long(df[upper_col].shift(1).values, df['close'].shift(1).values, df['ATR'].shift(1).values).astype('int8')
                    df.loc[start_idx:, f'{strategy}_short_{config_id}'] = check_breakout_short(df[lower_col].shift(1).values, df['close'].shift(1).values, df['ATR'].shift(1).values).astype('int8')
                    
                    strategy = 'Fakeout'
                    # check_fakeout_long(prev_rates, current_sr_buy_decision):
                    # rates = last 4 candles, current_sr_buy_decision = SR buy decision
                    # Define the number of candles to look back
                    lookback = 4

                    # Apply the function using a rolling window
                    df[f'{strategy}_long_{config_id}'] = df.rolling(window=lookback).apply(
                        lambda window: check_fakeout_long(window, window['current_sr_buy_decision'].iloc[-1]),
                        raw=False
                    ).astype('int8')
                    df[f'{strategy}_short_{config_id}'] = df.rolling(window=lookback).apply(
                        lambda window: check_fakeout_short(window, window['current_sr_sell_decision'].iloc[-1]),
                        raw=False
                    ).astype('int8')

    # Bars Trend
    # check_bars_trend_long(prev_rates): - prev_rates = will be called with last 20, 50 and 100 rates
    # to send last 20, 50 and 100 rates, we will use the rolling window function
    for period in [20, 50, 100]:
        df[f'Bars_Trend_long_{period}'] = df.rolling(window=period).apply(
            lambda window: check_bars_trend_long(window),
            raw=False
        ).astype('int8')
        df[f'Bars_Trend_short_{period}'] = df.rolling(window=period).apply(
            lambda window: check_bars_trend_short(window),
            raw=False
        ).astype('int8')





                        


def calculate_all_targets(df, pip):
    pass
    #TODO: Implement this function
    feature_targets_bars = [5, 20, 50, 100]
    feature_targets_params = ['max_price_change', 'min_price_change', 'max_min_price_ratio','min_max_price_ratio' ,  'above_price_area', 'below_price_area' , 'above_below_price_area_ratio' , 'below_above_price_area_ratio']

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

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Get only relevant columns for the current timeframe
    relevant_cols = get_relevant_columns_by_timeframe(tf_name)

    # Ensure all required columns exist
    missing_cols = [col for col in relevant_cols if col not in df.columns]
    if missing_cols:
        nan_df = pd.DataFrame({c: np.nan for c in missing_cols}, index=df.index)
        df = pd.concat([df, nan_df], axis=1)

    # Indicators
    if tf_name in ['M15', 'M30', 'H1', 'H4', 'D1', 'W1']:
        print(f"    Calculating indicators for {symbol} {tf_name}, time is {datetime.now()}")
        calculate_all_indicators(df, pip)

    # SR
        print(f"    Calculating SR levels for {symbol} {tf_name}, time is {datetime.now()}")
        calculate_all_sr_levels(df)

    # Indicator_decisions
        print(f"    Calculating indicator decisions for {symbol} {tf_name}, time is {datetime.now()}")
        calculate_all_indicator_decisions(df)
    
    # Targets
        print(f"    Calculating targets for {symbol} {tf_name}, time is {datetime.now()}")
        calculate_all_targets(df, pip)

    # Patterns
    print(f"    Calculating patterns for {symbol} {tf_name}, time is {datetime.now()}")
    if tf_name in ['M1', 'M5']:
        calculate_all_basic_patterns(df)
    else:
        calculate_all_candle_patterns(df)

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

    # If you want multiprocessing, uncomment and adjust:
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