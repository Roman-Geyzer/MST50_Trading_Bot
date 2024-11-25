# strategy.py
"""
This module contains the Strategy class, which represents a trading strategy.
The Strategy class manages open trades for a specific strategy, including entering positions, 
monitoring open trades, checking if trades are still open or closed, documenting closed trades, and performing actions like trailing stops or closing trades based on conditions.
Classes:
    Strategy: A class to represent a trading strategy.
Functions:
    initialize_strategies: Initialize all strategy instances based on the provided configuration.
    get_open_trades_from_terminal: Get all open trades from the terminal by magic number.
    handle_new_bar: Handle the arrival of a new bar.
    handle_new_minute: Handle the arrival of a new minute.
    document_closed_trade: Document a closed trade.
    check_open_trades: Monitor open trades and update their status.
    write_strategy_performance_file: Write the strategy's performance to a file.
    check_trail_active: Check if trailing is active for the strategy.
    check_trading_filters: Check the trading filters for a symbol and decide whether to continue checking for entry signals and indicators.
    check_and_place_orders: Check for trading signals and enter positions accordingly.
    close_all_trades: Close all trades in a specified direction for a given symbol.
    fill_request_data: Fill the request data for a trade operation.
    prep_and_order: Prepare and order.
    prep_and_close: Prepare and close.
    prep_and_update: Prepare and update.
"""

import pandas as pd
from datetime import datetime
import time
import os

# Determine if we are in backtesting mode
BACKTEST_MODE = os.environ.get('BACKTEST_MODE', 'False') == 'True'

# Import utility functions and constants
from .utils import (load_config,  get_final_magic_number, get_timeframe_string,
                    print_hashtaged_msg, attempt_with_stages_and_delay,print_with_info)
from .orders import calculate_lot_size, calculate_sl_tp, calculate_trail, get_mt5_trade_type, get_trade_direction, calculate_fast_trail, calculate_breakeven
from .indicators import Indicators
from .constants import DEVIATION, TRADE_DIRECTION
from .signals import RSISignal
from .candles import CandlePatterns
from .plotting import plot_bars

from .mt5_interface import (ORDER_TYPES, TRADE_ACTIONS, TIMEFRAMES, ORDER_TIME, ORDER_FILLING, TRADE_RETCODES,
                         positions_get, order_send, symbol_info_tick, symbol_info, symbol_select, last_error, history_deals_get, time_current) 

min_pips_for_trail_update = 2

drive = "x:" if os.name == 'nt' else "/Volumes/TM"

class Strategy:
    """
    A class to represent a trading strategy.

    This class manages open trades for a specific strategy, including entering positions,
    monitoring open trades, checking if trades are still open or closed, documenting closed trades,
    and performing actions like trailing stops or closing trades based on conditions.
    """

    def __init__(self, strategy_config):
        """
        Initialize the Strategy instance.

        Parameters:
            strategy_config (dict): A dictionary containing the strategy's configuration parameters.
        """
        self.config = strategy_config
        self.symbols = strategy_config['symbols']  # List of symbols to trade
        self.timeframe = strategy_config['timeframe']
        self.str_timeframe = get_timeframe_string(self.timeframe)
        self.magic_num = strategy_config['magic_num']
        self.strategy_num = strategy_config['strategy_num']
        self.strategy_name = strategy_config['strategy_name']
        self.tradeP_long = strategy_config['tradeP_long']
        self.tradeP_short = strategy_config['tradeP_short']
        self.trade_risk = strategy_config['tradeP_risk']
        self.fixed_order_size = strategy_config['tradeP_fixed_order_size']
        self.open_trades = self.get_open_trades_from_terminal()   # Dictionary to hold open trades: trade_id -> trade_info

        self.start_hour_trading = strategy_config['tradeP_hour_start']
        self.end_hour_trading = self.start_hour_trading + strategy_config['tradeP_hour_length']

        # Initialize required columns set
        self.required_columns = set(['time', 'open', 'high', 'low', 'close', 'spread'])

        # Add ATR and RSI columns (they are always needed)
        self.required_columns.update(['ATR', 'RSI'])

        # Initialize Indicators and collect required columns(in rates df) based on indicators
        self.init_indicators_and_collect_required_columns()

        self.candle_params = strategy_config['candle_params']
        # current_tf 
        self.current_tf_candle_patterns = CandlePatterns(self.candle_params.get('current_tf', {}))
        self.current_candle_patterns_active = self.current_tf_candle_patterns.check_candle_patterns_active()
        self.current_candle_patterns_history_length = max(self.current_tf_candle_patterns.pattern_candles_count, 4)

        # higher_tf
        self.higher_tf_candle_patterns =  CandlePatterns(self.candle_params.get('higher_tf', {}))
        self.higher_timeframe = strategy_config['candle_params']['higher_tf']['timeframe']
        # self.higher_str_timeframe = get_timeframe_string(self.higher_timeframe)
        self.higher_candle_patterns_active = self.higher_tf_candle_patterns.check_candle_patterns_active()
        self.higher_candle_patterns_history_length = max(self.higher_tf_candle_patterns.pattern_candles_count, 4)

        # lower_tf
        self.lower_tf_candle_patterns = CandlePatterns(self.candle_params.get('lower_tf', {}))
        self.lower_timeframe = strategy_config['candle_params']['lower_tf']['timeframe']
        # self.lower_str_timeframe = get_timeframe_string(self.lower_timeframe)
        self.lower_candle_patterns_active = self.lower_tf_candle_patterns.check_candle_patterns_active()
        self.lower_candle_patterns_history_length = max(self.lower_tf_candle_patterns.pattern_candles_count, 4)
        
        self.candle_patterns_active = (
            self.current_candle_patterns_active or
            self.higher_candle_patterns_active or
            self.lower_candle_patterns_active
        )
        #future implementation: add more signals
        #self.er_signal = ERSignal(strategy_config['exitP_ER_low_value'], strategy_config['EXITP_ER_high_value'])

        self.sl_method = strategy_config['sl_method']
        self.sl_param = strategy_config['sl_param']
        if self.sl_method in {'UseCandles_SL'}:
            self.sl_param = int(self.sl_param)

        self.tp_method = strategy_config['tp_method']
        self.tp_param = strategy_config['tp_param']
        if self.tp_method in {'UseCandles_TP'}:
            self.tp_param = int(self.tp_param)
        #TODO: update the exit params to be part of the class to avoid using get from dict
        self.exit_params = strategy_config.get('exit_params', {})
        self.daily_candle_exit_hour = self.exit_params.get('exitP_daily_candle_exit_hour', 0)

        self.trail_both_directions = strategy_config['trail_params']['trail_both_directions']

        # Trailing stop parameters
        self.trail_method = strategy_config['trail_params']['trail_method']
        self.trail_param = strategy_config['trail_params']['trail_param']
        # Fast trail parameters
        self.use_fast_trail = strategy_config['trail_params']['use_fast_trail']
        self.fast_trail_minutes_count = strategy_config['trail_params']['fast_trail_minutes_count']
        self.fast_trail_ATR_start_multiplier = strategy_config['trail_params']['fast_trail_ATR_start_multiplier']
        self.fast_trail_ATR_trail_multiplier = strategy_config['trail_params']['fast_trail_ATR_trail_multiplier']
        # Move to breakeven parameters
        self.use_move_to_breakeven = strategy_config['trail_params']['use_move_to_breakeven']
        self.breakeven_ATRs = strategy_config['trail_params']['breakeven_ATRs']
        # Check if trailing is enabled
        self.trail_enabled = self.trail_method or self.use_fast_trail or self.use_move_to_breakeven

        self.backtest_params = strategy_config['backtest_params']
        self.backtest_start_date = strategy_config['backtest_params']['backtest_start_date']
        self.backtest_tf = strategy_config['backtest_params']['backtest_tf']
        # self.backtest_str_tf = get_timeframe_string(self.backtest_tf)

        self.rsi_signal = RSISignal(
            rsi_period=self.config.get('filterP_rsi_period', 14),
            max_deviation=self.config['filterP_max_rsi_deviation'],
            min_deviation=self.config['filterP_min_rsi_deviation']
        )

        
        # Define the documentation directory with strategy number and name
        self.documentation_dir = os.path.join(drive, 'documentation', f"strategy_{self.strategy_num}_{self.strategy_name}")

        # Create the documentation directory if it doesn't exist
        if not os.path.exists(self.documentation_dir):
            os.makedirs(self.documentation_dir)

        # Define the documented trades file path
        self.documanted_trades_file = os.path.join(self.documentation_dir, f"trades.csv")
        if not os.path.exists(self.documanted_trades_file):
            with open(self.documanted_trades_file, 'w') as f:
                # Header of file
                f.write("ticket,magic,symbol,direction,volume,price,sl,tp,open_daytime,close_datetime,trading_costs,profit,close_method,comment\n")
        
        # Define the general documentation file path
        self.documatation_performance_file = os.path.join(self.documentation_dir, f"performance.csv")
        if not os.path.exists(self.documatation_performance_file):
            with open(self.documatation_performance_file, 'w') as f:
                # Header of file
                f.write("date,hour,open_trades,margin,balance,margin_level,equity,profit\n")

        self.documatation_logs_file = os.path.join(self.documentation_dir, f"logs.csv")
        if not os.path.exists(self.documatation_logs_file):
            with open(self.documatation_logs_file, 'w') as f:
                #TODO: update the header for logs
                pass

        self.documatation_errors_file = os.path.join(self.documentation_dir, f"errors.csv")
        if not os.path.exists(self.documatation_errors_file):
            with open(self.documatation_errors_file, 'w') as f:
                #TODO: update the header for errors
                pass
        
    def init_indicators_and_collect_required_columns(self):
        """
        Initialize indicator and collect the required columns based on the indicators used in the strategy.
        """
        # First Indicator
        first_indicator_config = self.config['indicators']['first_indicator']
        if first_indicator_config['indicator_name']:
            self.first_indicator = Indicators(first_indicator_config)
            self.first_indicator_name = first_indicator_config['indicator_name']
            self.add_indicator_columns(first_indicator_config)
            if first_indicator_config['indicator_use'] in ['Enter', 'Both']:
                self.first_indicator_enter = True
            else:
                self.first_indicator_enter = False
            if first_indicator_config['indicator_use'] in ['Exit', 'Both']:
                self.first_indicator_exit = True
            else:
                self.first_indicator_exit = False
        else: # No indicator
            self.first_indicator = None
            self.first_indicator_enter = False
            self.first_indicator_exit = False

        # Second Indicator
        second_indicator_config = self.config['indicators']['second_indicator']
        if second_indicator_config['indicator_name']:
            self.second_indicator = Indicators(second_indicator_config)
            self.second_indicator_name = second_indicator_config['indicator_name']
            self.add_indicator_columns(second_indicator_config)
            if second_indicator_config['indicator_use'] in ['Enter', 'Both']:
                self.second_indicator_enter = True
            else:
                self.second_indicator_enter = False
            if second_indicator_config['indicator_use'] in ['Exit', 'Both']:
                self.second_indicator_exit = True
            if second_indicator_config['indicator_use'] in ['Exit', 'Both']:
                self.second_indicator_exit = True
            else:
                self.second_indicator_exit = False
        else:
            self.second_indicator = None
            self.second_indicator_enter = False
            self.second_indicator_exit = False

        # Third Indicator
        third_indicator_config = self.config['indicators']['third_indicator']
        if third_indicator_config['indicator_name']:
            self.third_indicator = Indicators(third_indicator_config)
            self.third_indicator_name = third_indicator_config['indicator_name']
            self.add_indicator_columns(third_indicator_config)
            if third_indicator_config['indicator_use'] in ['Enter', 'Both']:
                self.third_indicator_enter = True
            else:
                self.third_indicator_enter = False
            if third_indicator_config['indicator_use'] in ['Exit', 'Both']:
                self.third_indicator_exit = True
            else:
                self.third_indicator_exit = False
        else:
            self.third_indicator = None
            self.third_indicator_enter = False
            self.third_indicator_exit = False

    def add_indicator_columns(self, indicator_config):
        """
        Add required columns based on the indicator configuration.

        Parameters:
            indicator_config (dict): Configuration of the indicator.
        """
        indicator_name = indicator_config['indicator_name']
        indicator_params = indicator_config['indicator_params']

        if indicator_name == 'BB':
            # For Bollinger Bands, we need to add columns based on the deviation
            deviation = indicator_params.get('d', None)  # Assuming 'd' is the deviation parameter
            if deviation:
                deviation_str = str(int(float(deviation) * 10))
                self.required_columns.update({
                    f'BB{deviation_str}_Upper',
                    f'BB{deviation_str}_Middle',
                    f'BB{deviation_str}_Lower',
                    f'BB{deviation_str}_Bool_Above',
                    f'BB{deviation_str}_Bool_Below',
                })
        elif indicator_name == 'MA':
            # For Moving Averages, if present, load all MA columns
            self.required_columns.update({'MA_7', 'MA_21', 'MA_50', 'MA_200'})
            self.required_columns.update({'MA_7_comp', 'MA_21_comp', 'MA_50_comp'})
        elif indicator_name == 'GA':
            # For Guppy Averages, load all GA columns
            self.required_columns.update({'GA_50', 'GA_100', 'GA_200', 'GA_500'})
        else:
            pass
            # place holder for other indicators


    def get_open_trades_from_terminal(self):
        """
        Get all open trades from the terminal by magic number.
        Returns:
            dict: Dictionary containing open trades by ticket number.
        """
        open_trades = {}
        if BACKTEST_MODE:
            return open_trades # Exit the method early if in backtesting mode - no open trades on strat of backtest
        all_positions = positions_get()
        if all_positions is None:
            return open_trades
        compare_magic = self.magic_num // 1000
        for position in all_positions:
            if position['magic'] // 1000 == compare_magic:
                open_trades[position['ticket']] = position
        return open_trades



    @staticmethod            
    def initialize_strategies(strategies_run_mode = ['live']):
        """
        Initialize all strategy instances based on the provided configuration.
        Args:
            strategies_run_mode (list): List of modes in which the trading strategies can run.
        Returns:
            dict: Dictionary containing initialized strategy instances.
        """
        # Load the configuration based on the provided run mode
        strategies_config = load_config(strategies_run_mode=strategies_run_mode)
        # Print selected strategies and their settings
        for strategy_num , settings in strategies_config.items():print(f"""Executing strategy no. {strategy_num}
                                                                        strategy name: {settings['strategy_name']},
                                                                        setragy run mode: {settings['strategy_status']},
                                                                        strategy magic number: {settings['magic_num']},
                                                                        strategy status: {settings['strategy_status']},
                                                                        with symbols: {settings['symbols']} 
                                                                        timeframe: {settings['timeframe']},""")
        # Initialize all strategy instances
        strategies = {strategy_name: Strategy(strategy_config) for strategy_name, strategy_config in strategies_config.items()}

        #check all active strategies have valid timeframes:
        for strategy in strategies.values():
            if strategy.timeframe is None:
                print_hashtaged_msg(3, f"starategy no. {strategy.num}, name: {strategy.strategy_name} has an invalid timeframe and will be removed")
                strategies.pop(strategy.num)
            if strategy.higher_timeframe is not None and strategy.timeframe >= strategy.higher_timeframe:
                print_hashtaged_msg(3, f"starategy no. {strategy.num}, name: {strategy.strategy_name} has invalid timeframes and will be removed",
                                    f"higher timeframe: {strategy.higher_timeframe}, timeframe: {strategy.timeframe}")
            if strategy.lower_timeframe is not None and strategy.timeframe >= strategy.lower_timeframe:
                print_hashtaged_msg(3, f"starategy no. {strategy.num}, name: {strategy.strategy_name} has invalid timeframes and will be removed",
                                    f"lower timeframe: {strategy.lower_timeframe}, timeframe: {strategy.timeframe}")
        return strategies


    def handle_new_bar(self, symbols):
        """
        Handle the arrival of a new bar.
        This method is called when a new bar arrives in the strategy's timeframe.
        It is used update indicators, check for trading signals, etc.
        """
        #print(f"Handling new bar for strategy no. {self.strategy_num}, strategy name: {self.strategy_name}, strategy timeframe: {self.str_timeframe}")
        self.check_open_trades() # check all open trades to make sure they are still open
                                       # document the closed trades and update strategy performance file
                                       # update the stratgy class variables
        for stra_symbol in self.symbols:   
            self.check_and_place_orders(stra_symbol, symbols[stra_symbol]) # check for trading signals and place orders
            self.check_exit_conditions(stra_symbol, symbols[stra_symbol].get_tf_rates(self.timeframe)) # check exit conditions for open trades
            if self.trail_enabled:
                self.monitor_open_trades(stra_symbol, symbols[stra_symbol].get_tf_rates(self.timeframe)) # update SL, TP, etc. - runs every minute
    
    def handle_new_minute(self, symbols):
        """
        Handle the arrival of a new minute.
        This method is called when a new minute arrives in the strategy's timeframe.
        It is used to update indicators, check for trading signals, etc.
        """
        # for backtest puposes:
        if BACKTEST_MODE:
            tf = get_timeframe_string(self.timeframe)
        else:
            tf = 'M1'
        if not self.trail_enabled:
            return  # Exit the method early if trailing is not enabled

        self.check_open_trades() # check all open trades to make sure they are still open
                                        # document the closed trades and update strategy performance file   
        
        for stra_symbol in self.symbols:
            if symbols[stra_symbol].check_symbol_tf_flag(TIMEFRAMES[tf]): # Check if 1 minute data is available (for live trading)
                continue  # Skip to the next symbol if no 1 minute data is available
            self.monitor_open_trades(stra_symbol, symbols[stra_symbol].get_tf_rates(TIMEFRAMES[tf])) # update SL, TP, etc. - runs every minute


    def document_closed_trade(self, trade_id):
        """
        Document a closed trade.

        Parameters:
            trade_id (int): The ticket number of the trade.
            trade_info (dict): Information about the trade.
        """
        pass
        
        # TODO: comment and uncomment the following lines based on backtest optimization
        # TODO: validate in live trading
        trade_info = history_deals_get(None, None, trade_id)[0]
        if trade_info is None:
            print_hashtaged_msg(1, f"Trade {trade_id} no longer exists in MT5.")
            return
        else:
            with open(self.documanted_trades_file, 'a') as f:
                # Write the trade information to the file
                f.write(f"{trade_info['ticket']},{trade_info['magic']},{trade_info['symbol']},{trade_info['type']},{trade_info['volume']},\
                        {trade_info['price_open']},{trade_info['sl']},{trade_info['tp']},{trade_info['time']},{time_current()} ,\
                        {trade_info['swap']},{trade_info['profit']},{trade_info['reason']},{trade_info['comment']}\n")

    def check_open_trades(self):
        """
        Monitor open trades and update their status.

        Iterates over open trades to check if they are still open or have been closed.
        Updates trade records accordingly and handles closed trades.
        """
        # Loop through a copy of open_trades to avoid modification during iteration
        for trade_id, trade_info in list(self.open_trades.items()):
            # Get current position from MT5
            position = positions_get(ticket=trade_id)
            if position is None:
                # Trade is closed, handle closure
                self.open_trades.pop(trade_id)
                self.document_closed_trade(trade_id)
                
    
    def write_strategy_performance_file(self, account_info_dict):
        """
        Write the strategy's performance to a file.
        This method writes the strategy's performance metrics to a file for tracking and analysis.
        """
        with open(self.documatation_performance_file, 'a') as f:
            # Write the strategy's performance metrics to the file
            #TODO: comment or uncomment the following line based on backtest optimization
            f.write(f"{time_current().date()},{time_current().time()},{len(self.open_trades)},{account_info_dict['margin']},{account_info_dict['balance']},{account_info_dict['margin_level']},{account_info_dict['equity']},{account_info_dict['profit']}\n")

    def check_trading_filters(self):
        """
        Check the trading filters for a symbol and decide whether to continue checking for entry signals and indicators.
        """ 
        #check traing hours within the trading hours of the strategy
        current_hour = time_current().hour
        if current_hour < self.start_hour_trading or current_hour > self.end_hour_trading:
            return False # Exit the method early if not in trading hours
        current_day = str(time_current().weekday())
        
        if current_day not in self.config['tradeP_days']:
            return  False # Exit the method early if not a trading day
        
        # add more based on candles and other filters

        return True # Passed all filters - continue checking for entry signals and indicators in check_and_place_orders



    def check_and_place_orders(self, stra_symbol, symbol):
        """
        Check for trading signals and enter positions accordingly.
        
        Parameters:
            symbol (str): The symbol for which to check trading signals.
            rates_dict (dict): A dictionary containing historical rates for the symbol.
            each key is a timeframe and the value is a dataframe of rates for that timeframe
        """

        if not self.check_trading_filters():
            return  # Exit the method early if trading filters fail
        
        if symbol.check_symbol_tf_flag(self.timeframe):
            print_hashtaged_msg(1, f"No rates available for {stra_symbol} and timeframe {self.str_timeframe}")
            return  # Exit the method early if fetching fails

        rates = symbol.get_tf_rates(self.timeframe)
        tf_obj = symbol.get_tf_obj(self.timeframe)

        # Check candle patterns and make a trade decision
        candle_decision_set = {'both'} # Set to store candle decisions from different timeframes, 'both' is used as a flag to use indicator decision
        if self.candle_patterns_active:
            if self.current_candle_patterns_active:
                current_tf_candle_decision = self.current_tf_candle_patterns.make_trade_decision(rates, tf_obj)
                if not current_tf_candle_decision:
                    return # No trade if candle decision fails
                else:
                    candle_decision_set.add(current_tf_candle_decision)
            if self.higher_candle_patterns_active:
                higher_rates = symbol.get_tf_rates(self.higher_timeframe)
                higher_tf_candle_decision = self.higher_tf_candle_patterns.make_trade_decision(higher_rates, tf_obj)
                if not higher_tf_candle_decision:
                    return # No trade if candle decision fails
                else:
                    candle_decision_set.add(higher_tf_candle_decision)
                
            if self.lower_candle_patterns_active:
                lower_rates = symbol.get_tf_rates(self.lower_timeframe)
                lower_tf_candle_decision = self.lower_tf_candle_patterns.make_trade_decision(lower_rates, tf_obj)
                if not lower_tf_candle_decision:
                    return # No trade if candle decision fails
                else:
                    candle_decision_set.add(lower_tf_candle_decision)

        indicator_decision_set = {'both'} # Set to store indicators decisions from different indicators, 'both' is used as a flag to use candle decision
        # Proceed with normal indicator-based trade decision - up to 3 indicators can be used
        if self.first_indicator_enter: # Check if indicator is active
            first_indicator_decision, first_indicator_trade_data = self.first_indicator.make_trade_decision(rates)
            if first_indicator_decision is None:
                return
            else:
                indicator_decision_set.add(first_indicator_decision)
        if self.second_indicator_enter:
            second_indicator_decision, second_indicator_trade_data = self.second_indicator.make_trade_decision(rates)
            if second_indicator_decision is None:
                return
            else:
                indicator_decision_set.add(second_indicator_decision)
        if self.third_indicator_enter:
            third_indicator_decision, third_indicator_trade_data = self.third_indicator.make_trade_decision(rates)
            if third_indicator_decision is None:
                return
            else:
                indicator_decision_set.add(third_indicator_decision)
        
        # merge the candle and indicator decisions
        final_decision_set = candle_decision_set.union(indicator_decision_set)
        if len(final_decision_set) == 1: # No trade if no decision is made
            return
        final_decision_set.remove('both')
        if len(final_decision_set) == 2: # conflicting decisions - no trade
            return
        final_decision = final_decision_set.pop()

        # Check the RSI signal before making a trade
        if not self.rsi_signal.check_rsi_signal(rates['RSI'][-1], final_decision): 
            return  # No trade if RSI filter fails
        

        #TODO: implement ATR filter
        # check for ATR filter
    

        if final_decision == 'buy' and self.tradeP_long:
            self.close_all_trades(TRADE_DIRECTION.SELL, stra_symbol)
            if self.get_total_open_trades(stra_symbol) < self.config['tradeP_max_trades']: # Check if max trades reached
                self.place_order(TRADE_DIRECTION.BUY, stra_symbol, rates)
        elif final_decision == 'sell' and self.tradeP_short:
            self.close_all_trades(TRADE_DIRECTION.BUY, stra_symbol)
            if self.get_total_open_trades(stra_symbol) < self.config['tradeP_max_trades']: # Check if max trades reached
                self.place_order(TRADE_DIRECTION.SELL, stra_symbol, rates)

    def close_all_trades(self, direction, symbol):
        """
        Close all trades in a specified direction for a given symbol.
        -1 for sell trades, 1 for buy trades, 0 for all trades.
        """
        trades = [trade for trade in self.open_trades.values() if trade['symbol'] == symbol]
        if trades is None:
            return
        for trade in trades:
            position = positions_get(ticket=trade['ticket'])
            if position is None:
                print_hashtaged_msg(1, f"Trade {trade['ticket']} no longer exists in MT5.")
                self.document_closed_trade(self.open_trades.pop(trade['ticket']))
                continue
            position_direction = get_trade_direction(position['type'])
            if (TRADE_DIRECTION.BUY == position_direction and direction.value >= 0) or (TRADE_DIRECTION.SELL == position_direction and direction.value <= 0):
                self.close_trade_loop(position)



    def fill_request_data(self, direction, symbol, ticket, comment, rates):
        """     
        Fill the request data for a trade operation.
        Type of operation will be based on ticket value.
        if ticket > 0, it will be a close or update trade operation.
        if ticket < 0, it will be an open trade operation.
        """
        #TODO: update this method per the original mql5 code - need to use paremeters from the strategy config ???
        trade_type = get_mt5_trade_type(TRADE_DIRECTION(direction))
        price = self.get_price(symbol, direction)
        #round price to the nearest pip
        price = round(price, symbol_info(symbol)['digits'])
        if ticket > 0:
            position = positions_get(ticket=ticket)
            volume = position['volume']
            magic_num = position['magic']
        else:
            sl, tp = calculate_sl_tp(price, direction,self.config['sl_method'], self.config['sl_param'], self.config['tp_method'], self.config['tp_param'], symbol, rates)
            #pyro can't get np.float64
            if not BACKTEST_MODE:
                sl = float(sl)
                tp = float(tp)
            volume = calculate_lot_size(symbol, self.config['tradeP_risk'], self.fixed_order_size,sl)
            magic_num = get_final_magic_number(symbol, self.magic_num)

        request = {
            "action": TRADE_ACTIONS['DEAL'],
            "symbol": symbol,
            "volume": volume,  
            "type": trade_type,
            "price": price,
            "deviation": DEVIATION,
            "comment": comment,
        }

        if ticket > 0:  # close trade
            request["position"] = ticket
        else: # open trade
            request['magic'] = magic_num
            request["sl"] = sl
            request["tp"] = tp
        return request
    
    def prep_and_order(self, direction, symbol, ticket, comment, rates):
        """
        prepare and order
        used in order to retry the order in case of failure
        """
        request = self.fill_request_data(direction, symbol, ticket, comment, rates)
        result =  order_send(request)
        if result['retcode'] == TRADE_ACTIONS['DONE']:
            self.open_trades[result['order']] = result
            # Order succeeded, store trade info
            #print(f"Opened {direction} trade on {symbol}, ticket: {result['order']}")
        return result
    
    def prep_and_close(self, direction, symbol, ticket, comment):
        """
        prepare and close
        used in order to retry the order in case of failure
        """
        close_direction = -(direction.value) # Close the the position so need the opposite direction
        request = self.fill_request_data(close_direction, symbol, ticket, comment, rates=None)
        result =  order_send(request)
        return result
    
    def prep_and_update(self, trade_id, position, new_sl):
        """
        prepare and update trade by trade_id
        used in order to retry the order in case of failure
        """
        sym = position['symbol']
        sl = float(new_sl)
        tp = position['tp']
        type_order = position['type']
        action = TRADE_ACTIONS['SLTP'] # action for modify order
        order_time = ORDER_TIME['GTC']
        order_filling = ORDER_FILLING['FOK']

        if (sl != 0) and (tp == 0): 
            modify_order_request = {

                'action': action,
                'symbol':  sym,
                'position': trade_id ,
                'type': type_order,
                'sl': sl,
                'type_time': order_time,
                'type_filling': order_filling
                                    }
            return order_send(modify_order_request)

        elif (sl == 0) and (tp != 0): 
            modify_order_request = {

            'action': action,
            'symbol':  sym,
            'position': trade_id ,
            'type': type_order,
            'tp': tp,
            'type_time': order_time,
            'type_filling': order_filling
                                    }
            return order_send(modify_order_request)
        
        else:
            modify_order_request = {

            'action': action,
            'symbol':  sym,
            'position': trade_id ,
            'type': type_order,
            'tp': tp,
            'sl': sl,
            'type_time': order_time,
            'type_filling': order_filling
                                    }
            return order_send(modify_order_request)
    
    def close_trade_loop(self, position):
        """
        Close an open trade.

        Parameters:
            position (MT5 position object): The current position object from MT5.
        """

        # Prepare close request and send the order
        def check_return_func(result):
            if not result:
                return False
            return result['retcode'] == TRADE_RETCODES['DONE']
        ticket = position['ticket']
        symbol = position['symbol']
        direction = get_trade_direction(position['type'])
        loop_error_msg = f"Failed to close {direction} trade for {symbol}, strategy: {self.strategy_num}-{self.strategy_name}"
        comment = f"Close {direction.value}, strategy-{self.strategy_num}"

        result = attempt_with_stages_and_delay(5 , 3, 0.05, 1, loop_error_msg, check_return_func,
                                                    self.prep_and_close, (direction, symbol, ticket, comment))
        
        if not check_return_func(result):
            print_hashtaged_msg(1, f"Failed to close position {position['ticket']}, symbol: {position['symbol']}, strategy: {self.strategy_num}-{self.strategy_name}")
            print(f"mt5.last_error: {last_error()}")

        else:
            # Position closed successfully, remove from open trades
            self.open_trades.pop(position['ticket'])
            self.document_closed_trade(position['ticket'])

    def get_price(self, symbol, direction):
        """
        Retrun the current price for the symbol
        args:
            symbol (str): The symbol for which to prepare trade data.
            direction (int): The direction of the trade.
        returns:
            prict (float): The current price for the symbol.
        """

        if direction == TRADE_DIRECTION.BUY or direction == TRADE_DIRECTION.BUY.value:
            return symbol_info_tick(symbol)['ask']
        elif direction == TRADE_DIRECTION.SELL or direction == TRADE_DIRECTION.SELL.value:
            return symbol_info_tick(symbol)['bid']
        else:
            raise ValueError("Invalid trade direction")

    #TODO: add indicator trade data and implement logic when and how to use it
    def place_order(self, direction, symbol, rates):
        """
        Place an order for a given symbol and direction.

        Parameters:
            symbol (str): The trading symbol.
            direction Either DIRECTION_BUY or DIRECTION_SELL.
        """
        # Get symbol info
        symbol_i = symbol_info(symbol)
        if symbol_i is None:
            print(f"Symbol {symbol} not found")
            return

        # Ensure the symbol is visible in Market Watch
        if not symbol_i['visible']:
            if not symbol_select(symbol, True):
                print(f"Failed to select symbol {symbol}")
                return

        # Prepare order request and Send the order
        def check_return_func(result):
            if not result:
                return False
            return result['retcode'] == TRADE_ACTIONS['DONE']
        
        loop_error_msg = f"Failed to open {direction} trade for {symbol}, strategy: {self.strategy_num}-{self.strategy_name}"
        comment = f"{self.strategy_num}-{self.strategy_name}"

        result = attempt_with_stages_and_delay(4 , 5, 0.05, 1, loop_error_msg, check_return_func,
                                                    self.prep_and_order, (direction, symbol, -1, comment, rates))
        if not check_return_func(result):
            print_hashtaged_msg(1, f"Failed to open {direction} trade for {symbol}, strategy: {self.strategy_num}-{self.strategy_name}")
            print("mt5.last_error:", last_error())
        else:
            # Order succeeded, store trade info
            trade_info = {
                'symbol': symbol,
                'time': time.time(),
                'direction': direction,
                'ticket': result['order']
            }
            self.open_trades[result['order']] = trade_info
            # TODO: log the trade
            msg = f"Opened {direction} trade on {symbol}, ticket: {result['order']}"



    def check_exit_conditions(self, symbol, rates):
        """
        Check exit conditions for all open trades of the given symbol.
        If any exit condition is met, close the trade.

        Parameters:
            symbol (str): The trading symbol.
            rates (np.recarray): Rates data.
        """
        current_time = time_current()

        # Check for daily strategy - check exit conditions only at a specific hour
        if self.daily_candle_exit_hour:
            if self.daily_candle_exit_hour != current_time.hour:
                return # Exit the method early if not the exit hour (for other tf's than D1 we won't get here)
        if len(self.open_trades) == 0:
            return # Exit the method early if no open trades
        for trade_id, trade_info in list(self.open_trades.items()):
            if trade_info['symbol'] != symbol:
                continue  # Skip trades not related to the current symbol
            
            #TODO: optimize - move this to the relvent place in the code - only calculate bars in trade when needed
            # Calculate how long the trade has been open

            # Convert numpy.datetime64 to Python datetime
            trade_open_time = pd.Timestamp(trade_info['time']).to_pydatetime()
            time_diff = current_time - trade_open_time
            days_in_trade = time_diff.days
            bars_in_trade = self.calculate_bars_in_trade(trade_info, rates)

            # Retrieve trade details from MT5 to get current profit
            position = positions_get(ticket=trade_id)
            if position is None or len(position) == 0:
                print(f"Trade {trade_id} no longer exists in MT5.")
                self.open_trades.pop(trade_id, None)
                self.document_closed_trade(trade_id)
                continue

            profit = position['profit']

            #TODO: after updating the exit params to be part of the class, update the following conditions
            # Check daily profit close condition
            if self.exit_params.get('exitP_daily_profit_close', False):
                required_days = self.exit_params.get('exitP_daily_profit_close_days', 1)
                if days_in_trade >= required_days and profit > 0:
                    if self.exit_params.get('exitP_daily_close_hour', 22) == current_time.hour:
                        print(f"Closing trade {trade_id} due to daily profit close condition.")
                        self.close_trade(position)
                        continue  # Move to the next trade

            # Check daily close condition
            if self.exit_params.get('exitP_daily_close', False):
                required_days = self.exit_params.get('exitP_daily_close_days', 1)
                if days_in_trade >= required_days:
                    if self.exit_params.get('exitP_daily_close_hour', 22) == current_time.hour:
                        print(f"Closing trade {trade_id} due to daily close condition.")
                        self.close_trade(position)
                        continue  # Move to the next trade

            # Check bars close condition
            bars_close = self.exit_params.get('exitP_bars_close', 0)
            if bars_close > 0 and bars_in_trade >= bars_close:
                print(f"Closing trade {trade_id} due to bars close condition.")
                self.close_trade(position)
                continue  # Move to the next trade

            # check strategy close condition
            if self.first_indicator_exit:
                if self.first_indicator.check_exit_condition(rates, position['type']):
                    print(f"Closing trade {trade_id} due to first indicator: {self.first_indicator} exit condition.")
                    self.close_trade(position)
                    continue # Move to the next trade
            if self.second_indicator_exit:
                if self.second_indicator.check_exit_condition(rates, position['type']):
                    print(f"Closing trade {trade_id} due to second indicator: {self.second_indicator,} exit condition.")
                    self.close_trade(position)
                    continue
            if self.third_indicator_exit:
                if self.third_indicator.check_exit_condition(rates, position['type']):
                    print(f"Closing trade {trade_id} due to third indicator: {self.third_indicator} exit condition.")
                    self.close_trade(position)
                    continue
            
            if self.trail_enabled:
                self.monitor_open_trades(symbol, rates)


    def monitor_open_trades(self, symbol, rates_df):
        """
        Monitor open trades and update their Stop Loss (SL) and Take Profit (TP) based on trailing strategies.
        
        Parameters:
            symbol (str): The trading symbol.
            rates_df (pd.DataFrame): DataFrame containing historical rates.
        """

        for trade_id, trade_info in list(self.open_trades.items()):
            if trade_info['symbol'] != symbol:
                continue  # Skip trades not related to the current symbol

            position = positions_get(ticket=trade_id)
            if position is None or len(position) == 0:
                print(f"Trade {trade_id} no longer exists in MT5.")
                self.open_trades.pop(trade_id, None)
                self.document_closed_trade(trade_id)
                continue

            self.monitor_open_trade(symbol, rates_df, trade_id, position)


    def monitor_open_trade(self, symbol_str, rates, trade_id, position):
        """
        Monitor single open trade and update its Stop Loss (SL) and Take Profit (TP) based on trailing strategies.

        Parameters:
            symbol_str (str): The trading symbol.
            rates (np.recarray): Rates data.
            trade_id (int): The trade ID.
            position (dict): The position data.
        """
        tick = symbol_info_tick(symbol_str)
        direction = position['type']  
        if direction == 0:  # buy
            price = tick['ask']
        else:  # sell
            price = tick['bid']
        current_sl = position['sl']
        point = symbol_info(symbol_str)['point']
        sl_price = [position['sl']]
        if self.use_fast_trail:
            sl_price.append(calculate_fast_trail(price, current_sl, self.trail_both_directions, direction, self.fast_trail_minutes_count, 
                                            self.fast_trail_ATR_start_multiplier, self.fast_trail_ATR_trail_multiplier,point, rates))
        if self.use_move_to_breakeven:
            sl_price.append(calculate_breakeven(price, current_sl, direction,self.trail_both_directions, self.breakeven_ATRs,
                                                 position['price_open'], point, rates))
        if self.trail_method:
            sl_price.append(calculate_trail(
                price, current_sl, self.trail_both_directions, direction,
                self.trail_method, self.trail_param, symbol_str, point, rates
            ))
        # Choose the new SL by comparing the new SLs from different methods

        # remove NaN values
        sl_price = [sl for sl in sl_price if sl is not None]
        if direction == 0:  # buy
            new_sl = max(sl_price)
        else:  # sell
            new_sl = min(sl_price)
        if new_sl and abs(new_sl - current_sl) > 10 * min_pips_for_trail_update * point: # pip is 10* point
            self.update_trade(trade_id=trade_id, position=position, new_sl=new_sl)



    def update_trade(self, trade_id, position, new_sl=None):
        """
        Update an open trade's SL and/or TP.
        
        Parameters:
            trade_id (int): The ticket number of the trade.
            new_sl (float, optional): New Stop Loss price.
            new_tp (float, optional): New Take Profit price.
        """

        def check_return_func(result):
            if not result:
                return False
            return result['retcode'] == TRADE_RETCODES['DONE']
        
        result = attempt_with_stages_and_delay(2,2, 0.1, 1 , f"Order update failed for trade: {trade_id}, retrying...",
                                                     check_return_func, self.prep_and_update, (trade_id, position, new_sl))
        if result['retcode'] != TRADE_ACTIONS['DONE']:
            print_hashtaged_msg(1, f"Failed to update trade {trade_id}. Retcode: {result['retcode']}")
            print(f"mt5.last_error: {last_error()}")
            print(f"new SL is:" , new_sl)
            print(f"position data: ", position)
        else:
            pass
            print(f"Successfully updated trade {trade_id}. New SL: {new_sl}")

    def get_total_open_trades(self, symbol):
        """
        Get the total number of open trades for a given symbol.
        """
        trades = self.open_trades.values()
        total_trades = 0
        for trade in trades:
            if trade['symbol'] == symbol:
                total_trades += 1
        return total_trades



    
    def calculate_bars_in_trade(self, trade_info, rates):
        """
        Calculate the number of bars since the trade was opened.

        Parameters:
            trade_info (dict): Information about the trade.
            rates (np.recarray): Rates data.

        Returns:
            int: Number of bars since the trade was opened.
        """
        open_time = trade_info['time']
        trade_time = pd.to_datetime(open_time, unit='s')
        latest_bar_time = pd.to_datetime(rates['time'][-1], unit='s')
        time_diff = latest_bar_time - trade_time
        bar_duration = pd.Timedelta(minutes=1)  # Adjust based on timeframe if needed
        bars = int(time_diff / bar_duration)
        return bars


