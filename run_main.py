#run_main.py
"""
Main script to run the MST50 strategy. This script is used to run the strategy in live mode
"""

import os

run_modes = ['transfer']
# back_test, demo, live, on_hold, dev, optimize, transfer


# Determine if we are in backtesting mode
os.environ['BACKTEST_MODE'] = 'False'  # Change to 'True' when ready to backtest
BACKTEST_MODE = os.environ.get('BACKTEST_MODE', 'False') == 'True'

from MST50.main import main


if __name__ == "__main__":
    main(run_modes=run_modes,BACKTEST_MODE=BACKTEST_MODE)

