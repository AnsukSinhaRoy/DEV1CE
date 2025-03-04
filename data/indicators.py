#indicators.py
import talib
import config
import numpy as np

def add_indicators(df):
    df["EMA_slow"] = talib.EMA(df["close"], timeperiod=config.SLOW_EMA_PERIOD)
    df["EMA_fast"] = talib.EMA(df["close"], timeperiod=config.FAST_EMA_PERIOD)
    df["RSI"] = talib.RSI(df["close"], timeperiod=config.RSI_PERIOD)

    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
        df["close"], timeperiod=config.BOLLINGER_PERIOD, nbdevup=1.5, nbdevdn=1.5, matype=0
    )

    df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=config.ATR_PERIOD)

    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
        df["close"], fastperiod=config.MACD_FAST, slowperiod=config.MACD_SLOW, signalperiod=config.MACD_SIGNAL
    )

    df["slowk"], df["slowd"] = talib.STOCH(
        df["high"], df["low"], df["close"],
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )

    # --- Additional Indicators ---

    # 1. Normalized ATR (Volatility Measure)
    df["Norm_ATR"] = df["ATR"] / df["close"]

    # 2. Historical Volatility (Standard deviation of log returns)
    df["Log_Returns"] = np.log(df["close"] / df["close"].shift(1))
    df["Volatility"] = df["Log_Returns"].rolling(window=20).std()

    # 3. ADX (Trend Strength)
    df["ADX"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # 4. ROC (Momentum Indicator)
    df["ROC"] = talib.ROC(df["close"], timeperiod=14)

    # --- Default Columns ---
    df["signals"] = 0
    df["portfolio_value"] = 10000.0
    df["shares"] = 0.0

    return df