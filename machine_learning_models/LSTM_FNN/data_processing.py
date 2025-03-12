import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(features, df, window_size):
    
    """
    Preprocess the data and generate tensors for training and testing.

    Args:
        df (pd.DataFrame): The input dataframe containing the features.
        window_size (int): The size of the time-series window.

    Returns:
        X (np.array): Input sequences of shape [samples, timesteps, features].
        y (np.array): Target values of shape [samples].
        processed_df (pd.DataFrame): The processed dataframe after scaling.
        scaler (MinMaxScaler): The scaler object for inverse transformation.
    """
    df.dropna(inplace=True)

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])

    # Convert into time-series sequences
    X, y = [], []
    df_as_np = df[features].to_numpy()
    close_prices = df['close'].to_numpy()
    for i in range(len(df_as_np) - window_size):
        row = df_as_np[i:i+window_size]  # Shape: (window_size, num_features)
        X.append(row)
        y.append(close_prices[i+window_size])  # Target at t+window_size

    return np.array(X), np.array(y), df, scaler
