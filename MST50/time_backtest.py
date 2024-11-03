# time_backtest.py

"""
This module provides backtesting versions of TradeHour and TimeBar classes,
which use the simulated current time from the backtesting module.
"""

from .mt5_backtest import backtest
    
class TradeHour:
    def __init__(self, backtest):
        self.backtest = backtest
        self.current_time = backtest.current_time
        self.current_hour = -1
        self.current_day = -1

    def update_current_time(self):
        self.current_time = self.backtest.current_time

    # Rest of the class...

class TimeBar:
    def __init__(self, backtest):
        self.backtest = backtest
        self.current_time = backtest.current_time
        # Initialize other attributes

    def update_current_time(self):
        self.current_time = self.backtest.current_time

    def is_new_hour(self):
        """
        Determine if a new hour has started.

        Returns:
            bool: True if a new hour has started, False otherwise.
        """
        self.update_current_time()
        if self.current_hour != self.current_time.hour or self.current_day != self.current_time.day:
            self.current_hour = self.current_time.hour
            self.current_day = self.current_time.day
            return True
        return False

    def is_new_day(self):
        """
        Determine if a new day has started.

        Returns:
            bool: True if a new day has started, False otherwise.
        """
        self.update_current_time()
        if self.current_day != self.current_time.day:
            self.current_day = self.current_time.day
            return True
        return False

    def is_new_week(self):
        """
        Determine if a new week has started.

        Returns:
            bool: True if a new week has started, False otherwise.
        """
        self.update_current_time()
        current_week = self.current_time.isocalendar()[1]
        if self.current_week != current_week:
            self.current_week = current_week
            return True
        return False

class TimeBar:
    """
    A class to track and determine if a new bar has started for different timeframes
    during backtesting.
    """
    def __init__(self):
        """
        Initialize the TimeBar instance using backtest.current_time.
        """
        self.current_time = backtest.current_time
        self.M1 = -1
        self.M5 = -1
        self.M15 = -1
        self.M30 = -1
        self.H1 = -1
        self.H4 = -1
        self.D1 = -1
        self.W1 = -1
        self.current_bar = 'W1'  # Start with the highest timeframe

    def update_current_time(self):
        """
        Update the current time from the backtesting module.
        """
        self.current_time = backtest.current_time

    def update_tf_bar(self):
        """
        Checks the highest timeframe that has a new bar and updates the respective attributes.
        Returns the timeframe that had the latest update.
        """
        self.update_current_time()
        current_time = self.current_time
        current_week = current_time.isocalendar()[1]
        updated = False

        # Check for a new weekly bar
        if self.W1 != current_week:
            self.W1 = current_week
            self.current_bar = "W1"
            updated = True

        # Check for a new daily bar
        elif self.D1 != current_time.day:
            self.D1 = current_time.day
            self.current_bar = "D1"
            updated = True

        # Check for a new 4-hour bar
        elif self.H4 != current_time.hour // 4:
            self.H4 = current_time.hour // 4
            self.current_bar = "H4"
            updated = True

        # Check for a new hourly bar
        elif self.H1 != current_time.hour:
            self.H1 = current_time.hour
            self.current_bar = "H1"
            updated = True

        # Check for a new 30-minute bar
        elif self.M30 != current_time.minute // 30:
            self.M30 = current_time.minute // 30
            self.current_bar = "M30"
            updated = True

        # Check for a new 15-minute bar
        elif self.M15 != current_time.minute // 15:
            self.M15 = current_time.minute // 15
            self.current_bar = "M15"
            updated = True

        # Check for a new 5-minute bar
        elif self.M5 != current_time.minute // 5:
            self.M5 = current_time.minute // 5
            self.current_bar = "M5"
            updated = True

        # Check for a new 1-minute bar
        elif self.M1 != current_time.minute:
            self.M1 = current_time.minute
            self.current_bar = "M1"
            updated = True

        # If no timeframe has changed, keep current_bar as is

        return self.current_bar

    def is_new_bar(self, timeframe):
        """
        Determine if a new bar has started for the given timeframe.

        Args:
            timeframe (str): The timeframe to check.

        Returns:
            bool: True if a new bar has started, False otherwise.
        """
        # Check if the current_bar is equal or higher than the timeframe
        timeframe_list = ['W1', 'D1', 'H4', 'H1', 'M30', 'M15', 'M5', 'M1']
        current_bar_index = timeframe_list.index(self.current_bar)
        timeframe_index = timeframe_list.index(timeframe)
        return current_bar_index <= timeframe_index

    def check_last_minute_of_hour(self):
        """
        Check if it's the last minute of the hour.

        Returns:
            bool: True if it's the last minute of the hour, False otherwise.
        """
        self.update_current_time()
        return self.current_time.minute == 59