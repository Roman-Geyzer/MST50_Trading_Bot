# Backtest/main_backtest.py

"""
This module contains the main backtesting function that runs the backtesting loop.
The backtesting loop advances the simulation time and executes strategies using the on_minute function.
Functions:
    main_backtest: Main backtest function that initializes the backtest and runs the backtesting loop.
"""

from .mt5_backtest import initialize_backtest, run_backtest, shutdown
from ..symbols import Symbol

def main_backtest(strategies):
    """
    Main backtest function that initializes the backtest and runs the backtesting loop.
    Args:
        strategies (dict): Dictionary of strategy instances.
    """
    initialize_backtest(strategies)  # Initialize the global backtest instance
    symbols = Symbol.initialize_symbols(strategies)
    # Run backtesting loop
    run_backtest(strategies, symbols)
    # Shutdown Backtest - clean up resources
    shutdown()