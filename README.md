# MST50 Trading Bot

This bot uses Python to create a fully autonomous trading Expert Advisor (EA). The first version will run locally on a PC and use MetaTrader 5 (MT5) as its backend for trading operations and pulling price data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Modules and Files](#modules-and-files)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The MST50 Trading Bot is designed to automate trading strategies using Python and MetaTrader 5. It supports both live trading and backtesting modes, allowing users to test their strategies on historical data before deploying them in a live environment.

## Features

- **Automated Trading**: Execute trading strategies automatically.
- **MT5 Integration**: Use other script running MetaTrader 5 and provides API for trading operations and data access.
- **Backtesting**: Test strategies on historical data - use of custom made functions to silmulate the MetaTrader 5 and avoid using API for performance
- **Job Scheduling**: Schedule tasks to run at specific intervals.
- **Logging and Utilities**: Comprehensive logging and utility functions.

## Installation

### Prerequisites

- Python 3.8 or higher
- Python program integrated with MetaTrader 5 (running along MetaTrader 5) and providing API to connect to
- `pip` (Python package installer)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MST50_Trading_Bot.git
   cd MST50_Trading_Bot
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Bot

1. Set the `BACKTEST_MODE` environment variable in `main.py`:

   ```python
   os.environ['BACKTEST_MODE'] = 'True'  # Set to 'False' for live trading
   ```

2. Run the main script:
   ```bash
   python run_main.py
   ```

## Project Structure

```
MST50_Trading_Bot
├── MST50
│   ├── Backtest
│   │   ├── main_backtest.py
│   │   ├── mt5_backtest.py
│   │   ├── mt5_backtest_constants.py
│   │   └── time_backtest.py
│   ├── main.py
│   ├── strategy.py
│   ├── symbols.py
│   └── utils.py
├──

run_main.py


└──

requirements.txt


```

## Configuration

- **`main.py`**: Main entry point for executing trading strategies.
- **`Backtest`**: Contains modules for running backtests.
- **`strategy.py`**: Define trading strategies.
- **`symbols.py`**: Define trading symbols.
- **`utils.py`**: Utility functions for logging and printing messages.

## Modules and Files

### `main.py`

This script is the main entry point for executing trading strategies using the provided configuration.

- **Functions**:
  - `main()`: Main function to execute trading strategies in either live trading or backtesting mode.
- **Modules**:
  - `schedule`: Python job scheduling for running the `on_minute` function every minute.
  - `time`: Time access and conversions.
  - `mt5_interface`: Interface to MetaTrader 5 for trading and data access.
  - `strategy`: Module for defining trading strategies.
  - `symbols`: Module for defining trading symbols.
  - `Backtest.main_backtest`: Module for running backtesting.
  - `utils`: Utility functions for logging and printing messages.
  - `run_bot`: Module for running the `on_minute` function.
- **Constants**:
  - `run_mode (list)`: Specifies the modes in which the trading strategies can run, either 'live' or 'demo'.
  - `cores (int)`: Number of cores to use for parallel processing.
  - `pytest_count (int)`: Number of times the pytest module has been run.

### `Backtest/main_backtest.py`

- **Functions**:
  - `run_backtest_loop`: Run the backtesting loop, advancing the simulation time and executing strategies.
  - `main_backtest`: Main backtest function that initializes the backtest and runs the backtesting loop.

### `Backtest/mt5_backtest.py`

- **Functions**:
  - `load_data`: Load historical data from CSV files into `symbols_data`.
  - `run_backtest`: Execute the backtest using the loaded data and strategies.
  - `initialize_backtest`: Initialize the backtest environment.
  - `shutdown`: Clean up resources after backtesting.

### `Backtest/mt5_backtest_constants.py`

- **Constants**:
  - Various constants used throughout the backtesting process, such as default values and configuration settings.

### `Backtest/time_backtest.py`

- **Classes**:
  - `TradeHour`: Class for managing trading hours.
  - `TimeBar`: Class for managing time bars in the backtest.

### `strategy.py`

- **Classes**:
  - `Strategy`: Base class for defining trading strategies.
- **Functions**:
  - `initialize_strategies`: Initialize and return a list of trading strategies based on the provided configuration.

### `symbols.py`

- **Classes**:
  - `Symbol`: Class for defining trading symbols and their properties.

### `utils.py`

- **Functions**:
  - `print_current_time`: Print the current time.
  - `print_hashtaged_msg`: Print a message with hashtags for formatting.
  - `load_config`: Load configuration settings from a file.
  - `safe_date_convert`: Safely convert date strings to datetime objects.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```

```
