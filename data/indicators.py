#indicators.py
import talib
class Indicators:
    @staticmethod
    def add_indicators(df):
        df["EMA_slow"] = talib.EMA(df["close"], timeperiod=75)
        df["EMA_fast"] = talib.EMA(df["close"], timeperiod=30)
        df["RSI"] = talib.RSI(df["close"], timeperiod=10)
        
        df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
            df["close"], timeperiod=15, nbdevup=1.5, nbdevdn=1.5, matype=0
        )
        
        df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=7)
        
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
            df["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        df["slowk"], df["slowd"] = talib.STOCH(
            df["high"], df["low"], df["close"],
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        return df
