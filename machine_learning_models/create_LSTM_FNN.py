import numpy as np
import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
parent_dir = r"E:\Python\GIT - DEV1CE\DEV1CE"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data.data_loader import DataLoader
from strategy.perfect_signal import generate_perfect_signals
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pandas as pd
import pandas as pd
from machine_learning_models.LSTM_FNN.data_processing import preprocess_data, split_data


dfmain = DataLoader.load_data()
df=dfmain[:67000]

df=generate_perfect_signals(df,30)

df.to_csv("machine_learning_models/LSTM_FNN/dfmain.csv", index=False)

features = [
    # Core OHLCV Data
    "open", "high", "low", "close", "volume",
    
    # Trend Indicators
    "EMA_slow", "EMA_fast", "RSI", "MACD", 
    "Ichimoku_Tenkan", "Ichimoku_Kijun", "Parabolic_SAR",

    # Volatility Indicators
    "ATR", "Norm_ATR", "BB_upper", "BB_middle", "BB_lower", "Volatility",

    # Momentum Indicators
    "ROC", "slowk", "slowd", "ADX",

    # Time Features
    "seconds" # Add more time features if available
]

# Preprocess the data
X, y, processed_df, scaler = preprocess_data(features, df, window_size=60)
print(f"Shape of Tensor X",  X.shape) 
print(f"Shape of Tensor y",  y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
