#config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "csv", "TRY.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "output", "FullData.csv")
SIGNALS_CSV = os.path.join(BASE_DIR, "output", "signals.csv")

# Trading Parameters
FAST_EMA_PERIOD = 50
SLOW_EMA_PERIOD = 100
RSI_PERIOD = 10
BOLLINGER_PERIOD = 15
ATR_PERIOD = 7
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOCH_K = 14
STOCH_D = 3
TRAILING_STOP_PCT = 0.02
Starting_Capital = 10000
