# calculate_indicator_decisions.py
"""
This module will calculates the indicator decisions for the given data.
Each function will recieve rates with lookback period and will return the trade decision.
Each function will return the trade decision and the trade parameters.
"""

import numpy as np

#####################################################################
#                                                                   #
#///////////////////////// Bollinger Bands /////////////////////////#
#                                                                   #
#####################################################################

def check_bb_with_long(prev_candle_bool_above):
    """
    Determine trade decision based on BB_With strategy using precomputed boolean columns.
    Buy if price is above the upper band.
    
    Parameters:
        prev_candle_bool_above (np.ndarray): Boolean array indicating if previous candle was above upper BB.
    
    Returns:
        np.ndarray: Boolean array for long decisions.
    """
    return prev_candle_bool_above.astype(bool)

def check_bb_with_short(prev_candle_bool_below):
    """
    Determine trade decision based on BB_With strategy using precomputed boolean columns.
    Sell if price is below the lower band.
    
    Parameters:
        prev_candle_bool_below (np.ndarray): Boolean array indicating if previous candle was below lower BB.
    
    Returns:
        np.ndarray: Boolean array for short decisions.
    """
    return prev_candle_bool_below.astype(bool)

def check_bb_return_long(prev_candle_bool_below, prev_prev_candle_bool_below):
    """
    Determine trade decision based on BB_Return strategy using precomputed boolean columns.
    Buy after price returns from below the lower band to above the lower band.
    
    Parameters:
        prev_candle_bool_below (np.ndarray): Boolean array indicating if previous candle was below lower BB.
        prev_prev_candle_bool_below (np.ndarray): Boolean array indicating if two candles ago were below lower BB.
    
    Returns:
        np.ndarray: Boolean array for return long decisions.
    """
    return np.logical_and(prev_prev_candle_bool_below, np.logical_not(prev_candle_bool_below))

def check_bb_return_short(prev_candle_bool_above, prev_prev_candle_bool_above):
    """
    Determine trade decision based on BB_Return strategy using precomputed boolean columns.
    Sell after price returns from above the upper band to below the upper band.
    
    Parameters:
        prev_candle_bool_above (np.ndarray): Boolean array indicating if previous candle was above upper BB.
        prev_prev_candle_bool_above (np.ndarray): Boolean array indicating if two candles ago were above upper BB.
    
    Returns:
        np.ndarray: Boolean array for return short decisions.
    """
    return np.logical_and(prev_prev_candle_bool_above, np.logical_not(prev_candle_bool_above))

def check_bb_over_long(prev_candle_close, prev_prev_candle_close, prev_middle_band, prev_prev_middle_band):
    """
    Determine trade decision based on BB_Over strategy using precomputed boolean columns.
    Buy if price crosses above the middle band from below.
    
    Parameters:
        prev_candle_close (np.ndarray): Previous candle's close prices.
        prev_prev_candle_close (np.ndarray): Two candles ago close prices.
        prev_middle_band (np.ndarray): Previous middle BB values.
        prev_prev_middle_band (np.ndarray): Two candles ago middle BB values.
    
    Returns:
        np.ndarray: Boolean array for over long decisions.
    """
    return np.logical_and(prev_candle_close > prev_middle_band, prev_prev_candle_close < prev_prev_middle_band)

def check_bb_over_short(prev_candle_close, prev_prev_candle_close, prev_middle_band, prev_prev_middle_band):
    """
    Determine trade decision based on BB_Over strategy using precomputed boolean columns.
    Sell if price crosses below the middle band from above.
    
    Parameters:
        prev_candle_close (np.ndarray): Previous candle's close prices.
        prev_prev_candle_close (np.ndarray): Two candles ago close prices.
        prev_middle_band (np.ndarray): Previous middle BB values.
        prev_prev_middle_band (np.ndarray): Two candles ago middle BB values.
    
    Returns:
        np.ndarray: Boolean array for over short decisions.
    """
    return np.logical_and(prev_candle_close < prev_middle_band, prev_prev_candle_close > prev_prev_middle_band)

#####################################################################
#                                                                   #
#///////////////////////// Moving Averages /////////////////////////#
#                                                                   #
#####################################################################

def check_ma_crossover_long(prev_short_ma, prev_prev_short_ma, prev_medium_ma, prev_prev_medium_ma, prev_long_ma):
    """
    Determine trade decision based on MA_Crossover strategy.
    Buy if short-term MA crosses above medium-term MA and medium-term MA is below long-term MA.
    
    Parameters:
        prev_short_ma (np.ndarray): Previous short-term MA values.
        prev_prev_short_ma (np.ndarray): Two candles ago short-term MA values.
        prev_medium_ma (np.ndarray): Previous medium-term MA values.
        prev_prev_medium_ma (np.ndarray): Two candles ago medium-term MA values.
        prev_long_ma (np.ndarray): Previous long-term MA values.
    
    Returns:
        np.ndarray: Boolean array for MA crossover long decisions.
    """
    return np.logical_and(
        np.logical_and(prev_short_ma > prev_medium_ma, prev_prev_short_ma < prev_prev_medium_ma),
        prev_medium_ma < prev_long_ma
    )

def check_ma_crossover_short(prev_short_ma, prev_prev_short_ma, prev_medium_ma, prev_prev_medium_ma, prev_long_ma):
    """
    Determine trade decision based on MA_Crossover strategy.
    Sell if short-term MA crosses below medium-term MA and medium-term MA is above long-term MA.
    
    Parameters:
        prev_short_ma (np.ndarray): Previous short-term MA values.
        prev_prev_short_ma (np.ndarray): Two candles ago short-term MA values.
        prev_medium_ma (np.ndarray): Previous medium-term MA values.
        prev_prev_medium_ma (np.ndarray): Two candles ago medium-term MA values.
        prev_long_ma (np.ndarray): Previous long-term MA values.
    
    Returns:
        np.ndarray: Boolean array for MA crossover short decisions.
    """
    return np.logical_and(
        np.logical_and(prev_short_ma < prev_medium_ma, prev_prev_short_ma > prev_prev_medium_ma),
        prev_medium_ma > prev_long_ma
    )

def check_price_ma_crossover_long(prev_candle_close, prev_prev_candle_close, prev_ma, prev_prev_ma):
    """
    Determine trade decision based on Price_MA_Crossover strategy.
    Buy if price crosses above MA from below.
    
    Parameters:
        prev_candle_close (np.ndarray): Previous candle's close prices.
        prev_prev_candle_close (np.ndarray): Two candles ago close prices.
        prev_ma (np.ndarray): Previous MA values.
        prev_prev_ma (np.ndarray): Two candles ago MA values.
    
    Returns:
        np.ndarray: Boolean array for price MA crossover long decisions.
    """
    return np.logical_and(prev_candle_close > prev_ma, prev_prev_candle_close < prev_prev_ma)

def check_price_ma_crossover_short(prev_candle_close, prev_prev_candle_close, prev_ma, prev_prev_ma):
    """
    Determine trade decision based on Price_MA_Crossover strategy.
    Sell if price crosses below MA from above.
    
    Parameters:
        prev_candle_close (np.ndarray): Previous candle's close prices.
        prev_prev_candle_close (np.ndarray): Two candles ago close prices.
        prev_ma (np.ndarray): Previous MA values.
        prev_prev_ma (np.ndarray): Two candles ago MA values.
    
    Returns:
        np.ndarray: Boolean array for price MA crossover short decisions.
    """
    return np.logical_and(prev_candle_close < prev_ma, prev_prev_candle_close > prev_prev_ma)

#####################################################################
#                                                                   #
#/////////////////////////////// Range /////////////////////////////#
#                                                                   #
#####################################################################

def check_sr_long(prev_lower_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on SR strategy using precomputed boolean columns.
    Buy if price is close to the lower SR level.
    
    Parameters:
        prev_lower_sr (np.ndarray): Previous lower SR levels.
        prev_close (np.ndarray): Previous close prices.
        prev_ATR (np.ndarray): Previous ATR values.
    
    Returns:
        np.ndarray: Boolean array for SR long decisions.
    """
    condition = np.logical_and(prev_lower_sr > 0, (prev_close - prev_lower_sr) < 2 * prev_ATR)
    return condition

def check_sr_short(prev_upper_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on SR strategy using precomputed boolean columns.
    Sell if price is close to the upper SR level.
    
    Parameters:
        prev_upper_sr (np.ndarray): Previous upper SR levels.
        prev_close (np.ndarray): Previous close prices.
        prev_ATR (np.ndarray): Previous ATR values.
    
    Returns:
        np.ndarray: Boolean array for SR short decisions.
    """
    condition = np.logical_and(prev_upper_sr > 0, (prev_upper_sr - prev_close) < 2 * prev_ATR)
    return condition

def check_breakout_long(prev_upper_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on Breakout strategy using precomputed boolean columns.
    Buy if price breaks above the upper SR level.
    
    Parameters:
        prev_upper_sr (np.ndarray): Previous upper SR levels.
        prev_close (np.ndarray): Previous close prices.
        prev_ATR (np.ndarray): Previous ATR values.
    
    Returns:
        np.ndarray: Boolean array for Breakout long decisions.
    """
    condition = np.logical_and(prev_upper_sr > 0, (prev_close - prev_upper_sr) > 0.1 * prev_ATR)
    return condition

def check_breakout_short(prev_lower_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on Breakout strategy using precomputed boolean columns.
    Sell if price breaks below the lower SR level.
    
    Parameters:
        prev_lower_sr (np.ndarray): Previous lower SR levels.
        prev_close (np.ndarray): Previous close prices.
        prev_ATR (np.ndarray): Previous ATR values.
    
    Returns:
        np.ndarray: Boolean array for Breakout short decisions.
    """
    condition = np.logical_and(prev_lower_sr > 0, (prev_lower_sr - prev_close) > 0.1 * prev_ATR)
    return condition

def check_fakeout_long(candle1low, candle2low, candle3low, candle4low, prev_lower_sr, current_sr_buy_decision, prev_ATR):
    """
    Vectorized function to determine Fakeout Long decisions.
    
    Parameters:
        candle1low (np.ndarray): Previous candle's low prices.
        candle2low (np.ndarray): Two candles ago low prices.
        candle3low (np.ndarray): Three candles ago low prices.
        candle4low (np.ndarray): Four candles ago low prices.
        prev_lower_sr (np.ndarray): Current lower SR levels.
        current_sr_buy_decision (np.ndarray): Current SR buy decisions.
        prev_ATR (np.ndarray): Previous ATR values.
    
    Returns:
        np.ndarray: Boolean array indicating Fakeout Long decisions.
    """
    fakeout_atr_slack = 0.5
    low_in_fakeout = np.minimum(candle1low, candle2low)
    low_before_fakeout = np.minimum(candle3low, candle4low)
    
    condition = (
        current_sr_buy_decision &                                  # Current SR buy decision is True
        (prev_lower_sr > 0) &                                       # Previous lower SR is greater than 0
        (low_in_fakeout <= (prev_lower_sr - fakeout_atr_slack * prev_ATR)) &  # Low in fakeout condition
        (low_before_fakeout >= prev_lower_sr)                      # Low before fakeout condition
    )
    
    return condition

def check_fakeout_short(candle1high, candle2high, candle3high, candle4high, prev_upper_sr, current_sr_sell_decision, prev_ATR):
    """
    Vectorized function to determine Fakeout Short decisions.
    
    Parameters:
        candle1high (np.ndarray): Previous candle's high prices.
        candle2high (np.ndarray): Two candles ago high prices.
        candle3high (np.ndarray): Three candles ago high prices.
        candle4high (np.ndarray): Four candles ago high prices.
        prev_upper_sr (np.ndarray): Current upper SR levels.
        current_sr_sell_decision (np.ndarray): Current SR sell decisions.
        prev_ATR (np.ndarray): Previous ATR values.
    """
    fakeout_atr_slack = 0.5
    high_in_fakeout = np.maximum(candle1high, candle2high)
    high_before_fakeout = np.maximum(candle3high, candle4high)
    
    condition = (
        current_sr_sell_decision &                                  # Current SR sell decision is True
        (prev_upper_sr > 0) &                                       # Previous upper SR is greater than 0
        (high_in_fakeout >= (prev_upper_sr + fakeout_atr_slack * prev_ATR)) &  # High in fakeout condition
        (high_before_fakeout <= prev_upper_sr)                      # High before fakeout condition
    )
    
    return condition

#####################################################################
#                                                                   #
#/////////////////////////////// Trend /////////////////////////////#
#                                                                   #
#####################################################################

def check_bars_trend_long(prev_rates_close):
    """
    Determine trade decision based on accepting reversal after a downtrend.
    
    Parameters:
        prev_rates_close (np.ndarray): Close prices of previous candles.
    
    Returns:
        float: 1.0 if current close < lowest close in the window (excluding current close), else 0.0
    """
    lowest_close = np.min(prev_rates_close[:-1])  # lowest close in the lookback period (excluding current close)
    current_close = prev_rates_close[-1]          # current close
    return 1.0 if current_close < lowest_close else 0.0 # will be converted to boolean on the caller side

def check_bars_trend_short(prev_rates_close):
    """
    Determine trade decision based on accepting reversal after an uptrend.
    
    Parameters:
        prev_rates_close (np.ndarray): Close prices of previous candles.
    
    Returns:
        float: 1.0 if current close > highest close in the window (excluding current close), else 0.0
    """
    highest_close = np.max(prev_rates_close[:-1])  # highest close in the lookback period (excluding current close)
    current_close = prev_rates_close[-1]           # current close
    return 1.0 if current_close > highest_close else 0.0 # will be converted to boolean on the caller side






#####################################################################
#                                                                   #
#////////////////////////////// GR /////////////////////////////////#
#                                                                   #
#####################################################################

# paremeter based - need to think if can be implemented is static file



#####################################################################
#                                                                   #
#///////////////////////////// RSI /////////////////////////////////#
#                                                                   #
#####################################################################

# highly baed on params - use or not overbought/oversold, also the look back period (for divergence - both start and end), consider for later
# in backtest call with 70/30, 80/20
def check_rsi_div_long():
    pass

def check_rsi_div_short():
    pass


def check_rsi_hid_div_long():
    pass

def check_rsi_hid_div_short():
    pass




#####################################################################
#                                                                   #
#///////////////////////////// Double //////////////////////////////#
#                                                                   #
#####################################################################

# highly baed on params - need to think if can be implemented is static file


#####################################################################
#                                                                   #
#/////////////////////////////// KAMA //////////////////////////////#
#                                                                   #
#####################################################################

# highly baed on params - need to think if can be implemented is static file



#####################################################################
#                                                                   #
#/////////////////////////////// Trend /////////////////////////////#
#                                                                   #
#####################################################################

# highly baed on params - need to think if can be implemented is static file