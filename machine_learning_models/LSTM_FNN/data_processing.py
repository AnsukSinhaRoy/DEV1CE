import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(features, df, window_size, lstm_output_features, future_steps=7):
    """
    Preprocess the data and generate tensors for training and testing.

    Args:
        features (list): List of feature names to use as input.
        df (pd.DataFrame): The input dataframe containing the features.
        window_size (int): The size of the time-series window.
        future_steps (int): Number of future steps to predict. Default is 7.
        lstm_output_features (list): List of feature names to predict. If None, all features are used.

    Returns:
        X (np.array): Input sequences of shape [samples, timesteps, features].
        y (np.array): Target values of shape [samples, future_steps, output_features].
        processed_df (pd.DataFrame): The processed dataframe after scaling.
        scaler (MinMaxScaler): The scaler object for inverse transformation.
    """

    print("Scaling data and generating tensors now....")
    
    df.dropna(inplace=True)

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])

    # Determine output feature indices
    if lstm_output_features is not None:
        output_indices = [lstm_output_features.index(f) for f in lstm_output_features]
    else:
        output_indices = slice(None)  # Use all features

    # Convert into time-series sequences
    X, y = [], []
    df_as_np = df[features].to_numpy()
    for i in range(len(df_as_np) - window_size - future_steps + 1):
        # Input sequence (past window_size steps)
        X.append(df_as_np[i:i+window_size])
        
        # Target sequence (next future_steps of specified output features)
        target_window = df_as_np[i+window_size : i+window_size+future_steps, output_indices]
        y.append(target_window)

    X = np.array(X)
    y = np.array(y)

    return X, y, df, scaler
