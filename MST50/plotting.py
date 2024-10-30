# plotting.py
"""
plotting.py module defines functions for plotting charts using OHLC data from a Timeframe instance and annotating trades if provided using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import kaleido # Required for static image export
from datetime import datetime
from .symbols import Timeframe

def plot_bars(timeframe_obj, trades=None, show=True, save_path=None):
    """
    Plots the bar chart using OHLC data from a Timeframe instance and annotates trades if provided.

    Parameters:
        timeframe_obj (Timeframe): Instance of the Timeframe class containing rates and metadata.
        trades (list of dict, optional): List of trade dictionaries with 'time', 'price', 'type' keys.
        show (bool, optional): If True, displays the chart.
        save_path (str, optional): If provided, saves the chart to the given path as an HTML file.
    """
    # Extract rates DataFrame, symbol, and timeframe from the Timeframe instance
    rates_df = timeframe_obj.get_rates()
    symbol = timeframe_obj.get_symbol_str()
    timeframe_str = timeframe_obj.get_timeframe_str()
    if rates_df is None or len(rates_df) == 0:
        print(f"No rates data available for {symbol}, {timeframe_str}.")
        return

    # Prepare the DataFrame
    df = rates_df.copy()
    df['Date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('Date', inplace=True)
    df = df[['open', 'high', 'low', 'close']]

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    )])

    # Add trade annotations
    if trades:
        for trade in trades:
            trade_time = datetime.fromtimestamp(trade['time'])
            trade_price = trade['price']
            trade_type = trade['type']
            if trade_type.lower() == 'buy':
                fig.add_trace(go.Scatter(
                    x=[trade_time],
                    y=[trade_price],
                    mode='markers',
                    marker=dict(symbol='triangle-up', color='green', size=12),
                    name='Buy',
                    showlegend=False
                ))
            elif trade_type.lower() == 'sell':
                fig.add_trace(go.Scatter(
                    x=[trade_time],
                    y=[trade_price],
                    mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=12),
                    name='Sell',
                    showlegend=False
                ))

    # Update layout
    fig.update_layout(
        title=f"{symbol} - {timeframe_str} Chart",
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'  # You can choose different templates like 'plotly', 'ggplot2', etc.
    )

    # Save or Show
    if save_path:
        # Plotly can save as HTML or static images (requires additional packages)
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    if show:
        fig.show()