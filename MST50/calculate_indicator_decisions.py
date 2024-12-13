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
    """
    return prev_candle_bool_above

def check_bb_with_short(prev_candle_bool_below):
    """
    Determine trade decision based on BB_With strategy using precomputed boolean columns.
    Sell if price is below the lower band.
    """
    return prev_candle_bool_below

def check_bb_return_long(prev_candle_bool_below, prev_prev_candle_bool_below):
    """
    Determine trade decision based on BB_Return strategy using precomputed boolean columns.
    Buy after price returns from below the lower band to above the lower band.
    """
    return prev_prev_candle_bool_below and not prev_candle_bool_below


def check_bb_return_short(prev_candle_bool_above, prev_prev_candle_bool_above):
    """
    Determine trade decision based on BB_Return strategy using precomputed boolean columns.
    Sell after price returns from above the upper band to below the upper band.
    """
    return prev_prev_candle_bool_above and not prev_candle_bool_above


def check_bb_over_long(prev_cndle_close, prev_prev_candle_close, prev_middle_band, prev_prev_middle_band):
    """
    Determine trade decision based on BB_Over strategy using precomputed boolean columns.
    Buy if price crosses above the middle band from below.
    """
    return prev_cndle_close > prev_middle_band and prev_prev_candle_close < prev_prev_middle_band

def check_bb_over_short(prev_cndle_close, prev_prev_candle_close, prev_middle_band, prev_prev_middle_band):
    """
    Determine trade decision based on BB_Over strategy using precomputed boolean columns.
    Sell if price crosses below the middle band from above.
    """
    return prev_cndle_close < prev_middle_band and prev_prev_candle_close > prev_prev_middle_band

#####################################################################
#                                                                   #
#///////////////////////// Moving Averages /////////////////////////#
#                                                                   #
#####################################################################


#TODO: need to call for multiple lookback periods (7vs21vs50, 21vs50vs200)
def check_ma_crossover_long(prev_short_ma, prev_prev_short_ma, prev_medium_ma,prev_prev_medium_ma, prev_long_ma  ):
    """
    Determine trade decision based on MA_Crossover strategy using precomputed boolean columns.
    Buy if short-term MA crosses above medium-term MA and medium-term MA is below long-term MA. - buy into strength after a pullback
    """
    return prev_short_ma > prev_medium_ma and prev_prev_short_ma < prev_prev_medium_ma and prev_medium_ma < prev_long_ma

def check_ma_crossover_short(prev_short_ma, prev_prev_short_ma, prev_medium_ma,prev_prev_medium_ma, prev_long_ma  ):
    """
    Determine trade decision based on MA_Crossover strategy using precomputed boolean columns.
    Sell if short-term MA crosses below medium-term MA and medium-term MA is above long-term MA. - sell into weakness after a rally
    """
    return prev_short_ma < prev_medium_ma and prev_prev_short_ma > prev_prev_medium_ma and prev_medium_ma > prev_long_ma



def check_price_ma_crossover_long(prev_candle_close, prev_prev_candle_close, prev_ma, prev_prev_ma):
    """
    Determine trade decision based on Price_MA_Crossover strategy using precomputed boolean columns.
    Buy if price crosses above MA from below.
    """
    return prev_candle_close > prev_ma and prev_prev_candle_close < prev_prev_ma

def check_price_ma_crossover_short(prev_candle_close, prev_prev_candle_close, prev_ma, prev_prev_ma):
    """
    Determine trade decision based on Price_MA_Crossover strategy using precomputed boolean columns.
    Sell if price crosses below MA from above.
    """
    return prev_candle_close < prev_ma and prev_prev_candle_close > prev_prev_ma



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
#/////////////////////////////// Range /////////////////////////////#
#                                                                   #
#####################################################################



# function will use default multiplier of 2 for ATR from SR level
def check_sr_long(prev_lower_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on SR strategy using precomputed boolean columns.
    Buy if price is close to the lower SR level.
    """
    if prev_lower_sr: # not 0
        if (prev_close - prev_lower_sr) < 2*prev_ATR: # buy if within 2*ATR i.e. close to SR level, side note: price can be below SR level
            return True
        
    return False

def check_sr_short(prev_upper_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on SR strategy using precomputed boolean columns.
    Sell if price is close to the upper SR level.
    """
    if prev_upper_sr: # not 0
        if (prev_upper_sr - prev_close) < 2*prev_ATR: # sell if within 2*ATR i.e. close to SR level, side note: price can be above SR level
            return True
        
    return False

# function will use default multiplier of 0.1 for ATR from SR level
def check_breakout_long(prev_upper_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on Breakout strategy using precomputed boolean columns.
    Buy if price breaks above the upper SR level.
    """
    if prev_upper_sr: # not 0
        if (prev_close - prev_upper_sr) > 0.1*prev_ATR: # buy if above 0.1*ATR i.e. breakout from SR level
            return True
        
    return False

def check_breakout_short(prev_lower_sr, prev_close, prev_ATR):
    """
    Determine trade decision based on Breakout strategy using precomputed boolean columns.
    Sell if price breaks below the lower SR level.
    """
    if prev_lower_sr: # not 0
        if (prev_lower_sr - prev_close) > 0.1*prev_ATR: # sell if above 0.1*ATR i.e. breakout from SR level
            return True
        
    return False

# function will use defaults: 2 candles from fakeout, 2 cadnles before fakeout and 0.5 multiplier for ATR (slack)
# with 2 and 2 default, it will be 4 candles lookback
def check_fakeout_long(prev_rates, current_sr_buy_decision):
    """
    Determine trade decision based on Fakeout strategy using precomputed boolean columns.
    Buy if price breaks below the lower SR level and then returns above it.
    """
    sr_level = prev_rates['lower_sr'][-1] # last candle lower sr level
    if sr_level: # not 0
        if current_sr_buy_decision:
            fakeout_atr_slack = 0.5
            bars_from_fakeout = 2
            # bars_before_fakeout = 2
            # total_lookback = bars_from_fakeout + bars_before_fakeout
             # Get lows for fakeout and previous period
            fakeout_lows = prev_rates['low'][-bars_from_fakeout:] # last 2 candles
            previous_lows = prev_rates['low'][:-bars_from_fakeout] # all candles before last 2 candles (in theoery it should be -total_lookback:-bars_from_fakeout)

            low_in_fakeout = np.min(fakeout_lows)
            low_before_fakeout = np.min(previous_lows)

            if low_in_fakeout <= sr_level - fakeout_atr_slack * prev_rates[-1]['ATR']:
                if low_before_fakeout >= sr_level:
                    return True
        
    return False

def check_fakeout_short(prev_rates, current_sr_sell_decision):
    """
    Determine trade decision based on Fakeout strategy using precomputed boolean columns.
    Sell if price breaks above the upper SR level and then returns below it.
    """
    sr_level = prev_rates['upper_sr'][-1] # last candle upper sr level
    if sr_level: # not 0
        if current_sr_sell_decision:
            fakeout_atr_slack = 0.5
            bars_from_fakeout = 2
            # bars_before_fakeout = 2
            # total_lookback = bars_from_fakeout + bars_before_fakeout
             # Get lows for fakeout and previous period
            fakeout_highs = prev_rates['high'][-bars_from_fakeout:] # last 2 candles
            previous_highs = prev_rates['high'][:-bars_from_fakeout] # all candles before last 2 candles (in theoery it should be -total_lookback:-bars_from_fakeout)

            high_in_fakeout = np.max(fakeout_highs)
            high_before_fakeout = np.max(previous_highs)

            if high_in_fakeout >= sr_level + fakeout_atr_slack * prev_rates[-1]['ATR']:
                if high_before_fakeout <= sr_level:
                    return True
        
    return False



#####################################################################
#                                                                   #
#/////////////////////////////// Trend /////////////////////////////#
#                                                                   #
#####################################################################

# highly baed on params - need to think if can be implemented is static file



#####################################################################
#                                                                   #
#//////////////////////////// Bars Trend ///////////////////////////#

# will call it with 20 and 100 lookback
def check_bars_trend_long(prev_rates):
    """
    Determine trade decision based on accepting reversal after a downtrend.
    """
    lowest_close = np.min(prev_rates['close'][:-1]) # lowest close in the lookback period (excluding current close)
    current_close = prev_rates['close'][-1] # current close
    if current_close < lowest_close: # if current close is lower than the lowest close in the lookback period - expect reversal
        return True

    return False

def check_bars_trend_short(prev_rates):
    """
    Determine trade decision based on accepting reversal after an uptrend.
    """
    highest_close = np.max(prev_rates['close'][:-1]) # highest close in the lookback period (excluding current close)
    current_close = prev_rates['close'][-1] # current close
    if current_close > highest_close: # if current close is higher than the highest close in the lookback period - expect reversal
        return True

    return False