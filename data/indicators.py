import talib
import config
import numpy as np
import pandas as pd

def add_indicators(df):
    """Compute technical indicators for trading strategy."""
    
    # --- Trend Indicators ---
    df["EMA_slow"] = talib.EMA(df["close"], timeperiod=config.SLOW_EMA_PERIOD)
    df["EMA_fast"] = talib.EMA(df["close"], timeperiod=config.FAST_EMA_PERIOD)
    df["RSI"] = talib.RSI(df["close"], timeperiod=config.RSI_PERIOD)

    # --- Volatility Indicators ---
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
        df["close"], timeperiod=config.BOLLINGER_PERIOD, nbdevup=1.5, nbdevdn=1.5, matype=0
    )
    
    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=config.ATR_PERIOD)
    df["Norm_ATR"] = df["ATR"] / df["close"]  # Normalized ATR for volatility

    # --- Momentum Indicators ---
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
        df["close"], fastperiod=config.MACD_FAST, slowperiod=config.MACD_SLOW, signalperiod=config.MACD_SIGNAL
    )

    df["slowk"], df["slowd"] = talib.STOCH(
        df["high"], df["low"], df["close"],
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )

    df["ROC"] = talib.ROC(df["close"], timeperiod=14)  # Rate of Change (Momentum)

    # --- Trend Strength Indicator ---
    df["ADX"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # --- Donchian Channel (Breakout Indicator) ---
    dc_length = 96
    df["DC_upper"] = df["high"].rolling(window=dc_length).max().shift(1)  # Highest high in 96 periods (avoid lookahead bias)
    df["DC_lower"] = df["low"].rolling(window=dc_length).min().shift(1)    # Lowest low in 96 periods
    df["DC_middle"] = (df["DC_upper"] + df["DC_lower"]) / 2       # Midpoint

    # --- Volume Analysis ---
    df["Volume_SMA"] = df["volume"].rolling(window=20).mean()  # Moving average of volume
    df["Volume_Threshold"] = df["Volume_SMA"] * 1.2  # White line (threshold for strong volume)
    
    # --- Historical Volatility ---
    df["Log_Returns"] = np.log(df["close"] / df["close"].shift(1))
    df["Volatility"] = df["Log_Returns"].rolling(window=20).std()

    # --- Default Columns ---
    df["signals"] = 0
    df["portfolio_value"] = 1000.0
    df["shares"] = 0.0

    return df
