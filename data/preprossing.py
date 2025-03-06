#preprocessing.py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd

def preprocess_data(df, window_size=60):
    """Prepare dataset for deep learning models."""

    # Drop missing values created by rolling window calculations
    df.dropna(inplace=True)
    
    # Select relevant features (drop non-numeric columns if needed)
    features = df.drop(columns=["signals", "portfolio_value", "shares", "Date"], errors="ignore") 

    # Scaling: MinMax for bounded features, StandardScaler for others
    min_max_cols = ["RSI", "slowk", "slowd", "ADX", "Close_Relative"]
    standard_cols = list(set(features.columns) - set(min_max_cols))

    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    df[min_max_cols] = min_max_scaler.fit_transform(df[min_max_cols])
    df[standard_cols] = standard_scaler.fit_transform(df[standard_cols])

    # Convert into time-series sequences (shape: [samples, timesteps, features])
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i+window_size].values)  # Last N timesteps
        y.append(df["signals"].iloc[i+window_size])  # Target trend (0 or 1)
    
    return np.array(X), np.array(y), min_max_scaler, standard_scaler
