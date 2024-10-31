# strategy.py


import pandas as pd
from datetime import datetime
import time
import os

# Import utility functions and constants
from .utils import (load_config,  get_final_magic_number, get_timeframe_string, print_with_info,
                    print_hashtaged_msg, attempt_i_times_with_s_seconds_delay, get_future_time)
from .orders import calculate_lot_size, calculate_sl_tp, calculate_trail, get_mt5_trade_data, get_trade_direction
from .indicators import Indicators
from .constants import DEVIATION, TRADE_DIRECTION
from .signals import RSISignal
from .candles import CandlePatterns
from .plotting import plot_bars

from .mt5_client import (ORDER_TYPES, TRADE_ACTIONS, TIMEFRAMES, ORDER_TIME, ORDER_FILLING, TRADE_RETCODES,
                         positions_get, order_send, symbol_info_tick, symbol_info, copy_rates,
                         history_deals_get, symbol_select, last_error) 

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
        self.open_trades = self.get_open_trades_from_terminal()   # Dictionary to hold open trades: trade_id -> trade_info
        self.indicator = Indicators(strategy_config)  # Instantiate the Indicators class
        self.rsi_signal = RSISignal(
            rsi_period=self.config.get('filterP_rsi_period', 14),
            max_deviation=self.config['filterP_max_rsi_deviation'],
            min_deviation=self.config['filterP_min_rsi_deviation']
        )

        self.candle_params = strategy_config['candle_params']
        # current_tf 
        self.current_tf_candle_patterns = CandlePatterns(self.candle_params.get('current_tf', {}))
        self.current_candle_patterns_active = self.current_tf_candle_patterns.check_candle_patterns_active()
        self.current_candle_patterns_history_length = max(self.current_tf_candle_patterns.pattern_candles_count, 4)

        # higher_tf
        self.higher_tf_candle_patterns =  CandlePatterns(self.candle_params.get('higher_tf', {}))
        self.higher_timeframe = strategy_config['candle_params']['higher_tf']['timeframe']
        self.higher_candle_patterns_active = self.higher_tf_candle_patterns.check_candle_patterns_active()
        self.higher_candle_patterns_history_length = max(self.higher_tf_candle_patterns.pattern_candles_count, 4)

        # lower_tf
        self.lower_tf_candle_patterns = CandlePatterns(self.candle_params.get('lower_tf', {}))
        self.lower_timeframe = strategy_config['candle_params']['lower_tf']['timeframe']
        self.lower_candle_patterns_active = self.lower_tf_candle_patterns.check_candle_patterns_active()
        self.lower_candle_patterns_history_length = max(self.lower_tf_candle_patterns.pattern_candles_count, 4)
        
        self.candle_patterns_active = (
            self.current_candle_patterns_active or
            self.higher_candle_patterns_active or
            self.lower_candle_patterns_active
        )
        #future implementation: add more signals
        #self.er_signal = ERSignal(strategy_config['exitP_ER_low_value'], strategy_config['EXITP_ER_high_value'])

        self.trade_method = strategy_config['tradeP_method']
        self.trade_limit_order_expiration_bars = strategy_config['tradeP_limit_order_expiration_bars']

        self.sl_method = strategy_config['sl_method']
        self.sl_param = strategy_config['sl_param']
        if self.sl_method in {'UseCandels_SL'}:
            self.sl_param = int(self.sl_param)

        self.tp_method = strategy_config['tp_method']
        self.tp_param = strategy_config['tp_param']
        if self.tp_method in {'UseCandels_TP'}:
            self.tp_param = int(self.tp_param)

        self.exit_params = strategy_config.get('exit_params', {})

        self.trail_method = strategy_config['trail_method']
        self.trail_param = strategy_config['trail_param']
        if self.trail_method in {'UseCandels_Trail_Close' , 'UseCandels_Trail_Extreme'}:
            self.trail_param = int(self.trail_param)
        self.trail_both_directions = strategy_config['trail_both_directions']
        self.trail_enabled = self.check_trail_active()




        
        # Define the documentation directory with strategy number and name
        self.documentation_dir = os.path.join(os.path.dirname(__file__), 'documentation', f"strategy_{self.strategy_num}_{self.strategy_name}")

        # Create the documentation directory if it doesn't exist
        if not os.path.exists(self.documentation_dir):
            os.makedirs(self.documentation_dir)

        # Define the documented trades file path
        self.documanted_trades_file = os.path.join(self.documentation_dir, f"strategy no.-_{self.strategy_num}, name: {self.strategy_name}_trades.csv")
        if not os.path.exists(self.documanted_trades_file):
            with open(self.documanted_trades_file, 'w') as f:
                # TODO: update the header to match the actual data
                # Header of file
                f.write("ticket,magic,symbol,direction,volume,price,sl,tp,time,open_daytime,close_datetime,trading_costs,profit,open_method,close_method,comment\n")
        
        # Define the general documentation file path
        self.documatation_performance_file = os.path.join(self.documentation_dir, f"strategy_{self.strategy_num}_performance.csv")
        if not os.path.exists(self.documatation_performance_file):
            with open(self.documatation_performance_file, 'w') as f:
                # TODO: update the header for performance analysis
                f.write("date,hour,open_trades,margin,balance,margin_level,equity,profit\n")
                # Define the general documentation file path

        self.documatation_logs_file = os.path.join(self.documentation_dir, f"strategy_{self.strategy_num}_logs.csv")
        if not os.path.exists(self.documatation_logs_file):
            with open(self.documatation_logs_file, 'w') as f:
                #TODO: update the header for logs
                pass

        self.documatation_errors_file = os.path.join(self.documentation_dir, f"strategy_{self.strategy_num}_errors.csv")
        if not os.path.exists(self.documatation_errors_file):
            with open(self.documatation_errors_file, 'w') as f:
                #TODO: update the header for errors
                pass
        
    def check_trail_active(self):
        """
        Check if trailing is active for the strategy.
        """
        problem =  self.trail_method is None or self.trail_method == 0 or self.trail_method == '0' or self.trail_method == "" or self.trail_param is None or self.trail_param <= 0
        return not problem
    

    def get_open_trades_from_terminal(self):
        """
        Get all open trades from the terminal by magic number.
        Returns:
            dict: Dictionary containing open trades by ticket number.
        """
        open_trades = {}
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
                                                                        strategy trade method: {settings['tradeP_method']},
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
        print(f"Handling new bar for strategy no. {self.strategy_num}, strategy name: {self.strategy_name}, strategy timeframe: {self.str_timeframe}")
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
        print(f"Handling new minute for strategy no. {self.strategy_num}")
        plot_bars(symbols.M1, trades=None, show=True, save_path='charts/chart1.png')
        if not self.trail_enabled:
            return  # Exit the method early if trailing is not enabled

        self.check_open_trades() # check all open trades to make sure they are still open
                                        # document the closed trades and update strategy performance file   
        
        for stra_symbol in self.symbols:
            if symbols[stra_symbol].check_symbol_tf_flag(TIMEFRAMES['M1']):
                continue  # Skip to the next symbol if no 1 minute data is available
            self.monitor_open_trades(stra_symbol, symbols[stra_symbol].get_tf_rates(TIMEFRAMES['M1'])) # update SL, TP, etc. - runs every minute


    def document_closed_trade(self, trade_id):
        """
        Document a closed trade.

        Parameters:
            trade_id (int): The ticket number of the trade.
            trade_info (dict): Information about the trade.
        """
        pass
        #trade_info = history_deals_get(position=trade_id)
        #TODO: implement error handling and logging if trade_info is None
        #TODO: implement logic to document closed trades


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
        #TODO: Implement logic to write strategy performance to a file

    def check_trading_filters(self, symbol, rates_df):
        """
        Check the trading filters for a symbol and decide whether to continue checking for entry signals and indicators.
        """ 
        #check traing hours within the trading hours of the strategy
        current_hour = datetime.now().hour
        start_hour = self.config['tradeP_hour_start']
        end_hour = start_hour + self.config['tradeP_hour_length']
        if current_hour < start_hour or current_hour > end_hour:
            return False # Exit the method early if not in trading hours
        current_day = str(datetime.now().weekday())
        
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
        if symbol.check_symbol_tf_flag(self.timeframe):
            print_hashtaged_msg(1, f"No rates available for {stra_symbol} and timeframe {self.str_timeframe}")
            return  # Exit the method early if fetching fails

        rates = symbol.get_tf_rates(self.timeframe)
        tf_obj = symbol.get_tf_obj(self.timeframe)

        if not self.check_trading_filters(symbol, rates):
            return  # Exit the method early if trading filters fail
        

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

        # Proceed with normal indicator-based trade decision
        if self.indicator.indicator_name != 0 : # Check if indicator is active
            indicator_decision, indicator_trade_data = self.indicator.make_trade_decision(rates, tf_obj)
            if indicator_decision is None: 
                return  # No trade if indicator fails
            if len(candle_decision_set) == 1:# Use indicator decision if candle decision is 'both' (the default value - which is a flag to use indicator decision)
                final_decision = indicator_decision
            else:
                candle_decision_set.remove('both')
                candle_decision = candle_decision_set.pop()
                if candle_decision == indicator_decision:
                    final_decision = indicator_decision
                else:
                    return  # No trade if candle and indicator decisions are conflicting
        else: # No indicator active - use candle decision
            if len(candle_decision_set) == 1:
                print_hashtaged_msg(2, f"No indicator active and candle decision is 'both' - double check the strategy configuration and logic")
                return # we should not be here!!! - No trade if no indicator and candle decision is 'both'
            candle_decision_set.remove('both')
            final_decision = candle_decision_set.pop()


        # Get the RSI values for filtering
        rsi_values = self.rsi_signal.calculate_rsi(rates)
        if not rsi_values:
            print_hashtaged_msg(1, f"Failed to calculate RSI for {stra_symbol} and timeframe {self.str_timeframe}")
            return  # Exit the method early if RSI values are not available
        
        # Check the RSI signal before making a trade
        if not self.rsi_signal.check_rsi_signal(rsi_values[-1],final_decision): 
            return  # No trade if RSI filter fails
    

        if final_decision == 'buy' and self.tradeP_long:
            self.close_all_trades(TRADE_DIRECTION.SELL, stra_symbol)
            if self.get_total_open_trades(stra_symbol) < self.config['tradeP_max_trades']: # Check if max trades reached
                self.place_order(TRADE_DIRECTION.BUY, stra_symbol)
        elif final_decision == 'sell' and self.tradeP_short:
            self.close_all_trades(TRADE_DIRECTION.BUY, stra_symbol)
            if self.get_total_open_trades(stra_symbol) < self.config['tradeP_max_trades']: # Check if max trades reached
                self.place_order(TRADE_DIRECTION.SELL, stra_symbol)

    def close_all_trades(self, direction, symbol):
        """
        Close all trades in a specified direction for a given symbol.
        -1 for sell trades, 1 for buy trades, 0 for all trades.
        """
        print_with_info( f"Closing all {direction} trades for {symbol}",levels_up=2)
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

    #TODO update this method to recive and updated per trade method

    def fill_request_data(self, direction, symbol, ticket, comment):
        """     
        Fill the request data for a trade operation.
        Type of operation will be based on ticket value.
        if ticket > 0, it will be a close or update trade operation.
        if ticket < 0, it will be an open trade operation.
        """
        #TODO: update this method per the original mql5 code - need to use paremeters from the strategy config
        trade_data = get_mt5_trade_data(TRADE_DIRECTION(direction) , trade_type=self.trade_method)
        price = self.get_price(symbol, direction)
        #round price to the nearest pip
        price = round(price, symbol_info(symbol)['digits'])
        if ticket > 0:
            position = positions_get(ticket=ticket)
            volume = position['volume']
            magic_num = position['magic']
        else:
            sl, tp = calculate_sl_tp(price, direction,self.config['sl_method'], self.config['sl_param'], self.config['tp_method'], self.config['tp_param'], symbol)
            volume = calculate_lot_size(symbol, self.config['tradeP_risk'],sl)
            magic_num = get_final_magic_number(symbol, self.magic_num)

        request = {
            "action": trade_data['action'],
            "symbol": symbol,
            "volume": volume,  
            "type": trade_data['type'],
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
            if trade_data['action'] == TRADE_ACTIONS['PENDING']:
                request["type_time"] = ORDER_TIME['GTC']
                request["expiration"] = get_future_time(symbol, self.timeframe, time.now(), self.trade_limit_order_expiration_bars)        #TimeCurrent(dt_struct) + PendingOrdersExpirationBars * PeriodSeconds(0);
        return request
    
    def prep_and_order(self, direction, symbol, ticket, comment):
        """
        prepare and order
        used in order to retry the order in case of failure
        """
        request = self.fill_request_data(direction, symbol, ticket, comment)
        result =  order_send(request)
        if result['retcode'] == TRADE_ACTIONS['DONE']:
            # Order succeeded, store trade info
            print(f"Opened {direction} trade on {symbol}, ticket: {result['order']}")
        return result
    
    def prep_and_close(self, direction, symbol, ticket, comment):
        """
        prepare and close
        used in order to retry the order in case of failure
        """
        close_direction = -(direction.value) # Close the the position so need the opposite direction
        request = self.fill_request_data(close_direction, symbol, ticket, comment)
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
            print_with_info(f"updating trade: modify_request_order: {modify_order_request}", levels_up=2)
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
            print_with_info(f"updating trade: modify_request_order: {modify_order_request}", levels_up=2)
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
            print_with_info(f"updating trade: modify_request_order: {modify_order_request}", levels_up=2)
            return order_send(modify_order_request)
    

    #TODO: update this method to collect data more methodically similar to place order
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

        result = attempt_i_times_with_s_seconds_delay(3, 0.05, loop_error_msg, check_return_func,
                                                    self.prep_and_close, (direction, symbol, ticket, comment))
        
        if not check_return_func(result):
            print_hashtaged_msg(1, f"Failed to close position {position['ticket']}, symbol: {position['symbol']}, strategy: {self.strategy_num}-{self.strategy_name}")
            print(f"mt5.last_error: {last_error()}")

        else:
            # Position closed successfully, remove from open trades
            print_with_info(f"Trade {position['ticket']} closed successfully.")
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
    def place_order(self, direction, symbol):
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

        result = attempt_i_times_with_s_seconds_delay(3, 0.05, loop_error_msg, check_return_func,
                                                    self.prep_and_order, (direction, symbol, -1, comment))
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
            msg = f"Opened {direction} trade on {symbol}, ticket: {result['order']}"
            print_with_info(msg)




    def check_exit_conditions(self, symbol, rates_df):
        """
        Check exit conditions for all open trades of the given symbol.
        If any exit condition is met, close the trade.
        
        Parameters:
            symbol (str): The trading symbol.
            rates_df (pd.DataFrame): DataFrame containing historical rates.
        """
        current_time = datetime.now()

        for trade_id, trade_info in list(self.open_trades.items()):
            if trade_info['symbol'] != symbol:
                continue  # Skip trades not related to the current symbol

            # Calculate how long the trade has been open
            open_time = datetime.fromtimestamp(trade_info['time'])
            time_diff = current_time - open_time
            days_in_trade = time_diff.days
            bars_in_trade = self.calculate_bars_in_trade(trade_info, rates_df)

            # Retrieve trade details from MT5 to get current profit
            position = positions_get(ticket=trade_id)
            if position is None or len(position) == 0:
                print(f"Trade {trade_id} no longer exists in MT5.")
                self.open_trades.pop(trade_id, None)
                self.document_closed_trade(trade_id)
                continue

            profit = position['profit']

            # Check daily profit close condition
            if self.exit_params.get('exitP_daily_profit_close', False):
                required_days = self.exit_params.get('exitP_daily_profit_close_days', 1)
                if days_in_trade >= required_days and profit > 0:
                    print(f"Closing trade {trade_id} due to daily profit close condition.")
                    self.close_trade(position)
                    continue  # Move to the next trade

            # Check daily close condition
            if self.exit_params.get('exitP_daily_close', False):
                required_days = self.exit_params.get('exitP_daily_close_days', 1)
                if days_in_trade >= required_days:
                    print(f"Closing trade {trade_id} due to daily close condition.")
                    self.close_trade(position)
                    continue  # Move to the next trade

            # Check bars close condition
            bars_close = self.exit_params.get('exitP_bars_close', 0)
            if bars_close > 0 and bars_in_trade >= bars_close:
                print(f"Closing trade {trade_id} due to bars close condition.")
                self.close_trade(position)
                continue  # Move to the next trade
            
            if self.trail_enabled:
                self.monitor_open_trades(symbol, rates_df)

    #TODO: add position to the new class for trades 
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


    def monitor_open_trade(self, symbol_str, rates_df, trade_id, position):
        """
        Monitor single open trade and update its Stop Loss (SL) and Take Profit (TP) based on trailing strategies.
        Parameters:
            symbol (str): The trading symbol.
            rates_df (pd.DataFrame): DataFrame containing historical rates.
        """
        # TODO: update after updating the calculate_trail
        tick = symbol_info_tick(symbol_str)
        direction = position['type']  # BUY are 0, 2, 4, 6, ; SELL are 1, 3, 5, 7
        if direction in {0, 2, 4, 6}:
            price = tick['ask']
        else:
            price = tick['bid']
        current_sl = position['sl']
        point = symbol_info(symbol_str)['point']

        new_sl = calculate_trail(price, current_sl, self.trail_both_directions, direction, self.trail_method, self.trail_param, symbol_str, point, rates_df)
        if new_sl and abs(new_sl - current_sl) > 10*point:
            self.update_trade(trade_id = trade_id, position = position, new_sl=new_sl)



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
        
        result = attempt_i_times_with_s_seconds_delay(3, 0.05, f"Order update failed for trade: {trade_id}, retrying...",
                                                     check_return_func, self.prep_and_update, (trade_id, position, new_sl))
        if result['retcode'] != TRADE_ACTIONS['DONE']:
            print_hashtaged_msg(1, f"Failed to update trade {trade_id}. Retcode: {result['retcode']}")
            print(f"mt5.last_error: {last_error()}")
        else:
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
        print_hashtaged_msg(1, f"Total open trades for {symbol}: {total_trades}")  
        return total_trades



    

    #TODO: update this method so the bars are part of the trade class and adds +1 each time the logic "touch" the trade - a lot more efficient
    def calculate_bars_in_trade(self, trade_info, rates_df):
        """
        Calculate the number of bars since the trade was opened.
        
        Parameters:
            trade_info (dict): Information about the trade.
            rates_df (pd.DataFrame): DataFrame containing historical rates.
        
        Returns:
            int: Number of bars since the trade was opened.
        """
        open_time = trade_info['time']
        trade_time = pd.to_datetime(open_time)
        latest_bar_time = pd.to_datetime(rates_df['time'].iloc[-1], unit='s')
        time_diff = latest_bar_time - trade_time
        bar_duration = pd.Timedelta(minutes=1)  # Assuming 1-minute bars; adjust based on timeframe
        bars = int(time_diff / bar_duration)
        return bars


