# data_loader.py
import pandas as pd
from config import DATA_PATH
import talib
import config
import numpy as np

class DataLoader:
    @staticmethod
    def load_data():
        # Load the data and parse the 'datetime' column
        df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])

        # Ensure the 'datetime' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        # Sort by datetime
        df.sort_values('datetime', inplace=True)

        # Extract the date from the datetime column
        df['date'] = df['datetime'].dt.date

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
        df["DC_upper"] = df["high"].rolling(window=dc_length).max().shift(1)  # Highest high in 96 periods
        df["DC_lower"] = df["low"].rolling(window=dc_length).min().shift(1)    # Lowest low in 96 periods
        df["DC_middle"] = (df["DC_upper"] + df["DC_lower"]) / 2       # Midpoint

        # --- Volume Analysis ---
        df["Volume_SMA"] = df["volume"].rolling(window=20).mean()  # Moving average of volume
        df["Volume_Threshold"] = df["Volume_SMA"] * 1.2  # Threshold for strong volume
        df["OBV"] = talib.OBV(df["close"], df["volume"])  # On-Balance Volume
        df["CMF"] = (df["close"] - df["low"] - (df["high"] - df["close"])) / (df["high"] - df["low"]) * df["volume"]
        df["CMF"] = df["CMF"].rolling(window=20).sum() / df["volume"].rolling(window=20).sum()  # Chaikin Money Flow

        # --- Historical Volatility ---
        df["Log_Returns"] = np.log(df["close"] / df["close"].shift(1))
        df["Cumulative_Returns"] = (df["close"] / df["close"].iloc[0]) - 1
        df["Volatility"] = df["Log_Returns"].rolling(window=20).std()

        # --- Relative Close Position ---
        df["Close_Relative"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # --- Ichimoku Cloud (Trend Confirmation) ---
        df["Ichimoku_Tenkan"] = (df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()) / 2
        df["Ichimoku_Kijun"] = (df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()) / 2
        df["Ichimoku_Senkou_A"] = ((df["Ichimoku_Tenkan"] + df["Ichimoku_Kijun"]) / 2).shift(26)
        df["Ichimoku_Senkou_B"] = ((df["high"].rolling(window=52).max() + df["low"].rolling(window=52).min()) / 2).shift(26)

        # --- Parabolic SAR (Trend Reversal) ---
        df["Parabolic_SAR"] = talib.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)

        # --- Sharpe Ratio (Risk-Adjusted Return) ---
        risk_free_rate = 0.01  # Assumed risk-free rate (modify based on market)
        df["Sharpe_Ratio"] = (df["Log_Returns"].rolling(window=20).mean() - risk_free_rate) / df["Log_Returns"].rolling(window=20).std()

        # --- Ulcer Index (Downside Risk) ---
        df["Drawdown"] = df["close"].rolling(window=14).max() - df["close"]
        df["Ulcer_Index"] = np.sqrt((df["Drawdown"] ** 2).rolling(window=14).mean())

        # --- Default Columns ---
        df["signals"] = 0
        df["portfolio_value"] = config.Starting_Capital
        df["shares"] = 0.0

        return df