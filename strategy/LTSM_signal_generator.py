import numpy as np
import pandas as pd
import os
import sys
from tensorflow.keras.models import load_model
from pathlib import Path
import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now, import your module
from data.data_loader import DataLoader



class LTSM:
    def __init__(self, df, model_path, features, window_size=50):
        self.df = df
        self.model_path = model_path
        self.features = features
        self.window_size = window_size
        self.model = load_model(model_path, compile=False)  # Load model without compiling

    def generate_sequential_signals(self):
        """
        Generate trading signals using the trained LSTM model.

        Returns:
            pd.DataFrame: Dataframe with added 'predicted_signal' column.
        """
        df = self.df.copy()
        df["predicted_signal"] = np.nan  # Placeholder for predictions

        for i in range(self.window_size, len(df)):
            past_data = df.iloc[i-self.window_size:i][self.features].values

            # Reshape for LSTM input (batch_size=1, timesteps=window_size, features)
            X_input = np.expand_dims(past_data, axis=0)

            # Predict trend (0 = downtrend, 1 = uptrend)
            pred_prob = self.model.predict(X_input)[0][0]
            print(pred_prob)
            pred_signal = int(pred_prob > 0.5)
            
            # Store the prediction
            df.loc[df.index[i], "predicted_signal"] = pred_signal


        return df

# Define the feature columns used for training
features = ["EMA_slow", "EMA_fast", "RSI", "BB_upper", "BB_middle", "BB_lower",
            "ATR", "MACD", "MACD_signal", "MACD_hist", "slowk", "slowd",
            "ROC", "ADX", "DC_upper", "DC_lower", "DC_middle", "Volume_SMA",
            "Volume_Threshold", "Volatility"]

# Load your dataset
df = DataLoader.load_data()


# Create LSTM object and generate signals
ltsm = LTSM(df, model_path="models/LSTM_model.h5", features=features)
df_with_signals = ltsm.generate_sequential_signals()

# Save or analyze results
df_with_signals.to_csv("signals_output.csv", index=False)
