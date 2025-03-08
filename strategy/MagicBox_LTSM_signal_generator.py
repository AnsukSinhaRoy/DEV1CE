import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
models_dir = parent_dir +"/models"
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True)

    def call(self, inputs):
        e = tf.keras.activations.tanh(tf.linalg.matmul(inputs, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)  # Softmax over time axis
        return Multiply()([inputs, a])  # Weighted sum
    
# Import custom modules
from data.data_loader import DataLoader
from visualization.plotter import Plotter


def generate_perfect_signals(df, lookahead):
    """Generate perfect trend signals based on future price movements."""
    print("Generating perfect trend signals...")
    df = df.copy()
    df['signals'] = np.nan  # Initialize with NaN for proper trend marking

    for i in range(len(df) - lookahead):
        future_prices = df['close'].iloc[i+1 : i+1+lookahead].values
        if len(future_prices) > 0:
            max_future_price = np.max(future_prices)
            min_future_price = np.min(future_prices)

            # If the price is near the lowest future price, mark an uptrend (1)
            if df['close'].iloc[i] <= min_future_price + 0.1:
                df.at[i, 'signals'] = 1  # Uptrend

            # If the price is near the highest future price, mark a downtrend (0)
            elif df['close'].iloc[i] >= max_future_price - 0.1:
                df.at[i, 'signals'] = 0  # Downtrend

    # **Propagate the trend signals forward**
    df['signals'] = df['signals'].ffill().fillna(0)  # Fill NaN values with previous trend, default to 0
    
    return df

def preprocess_data(df, window_size):
    """Prepare dataset for LSTM models with real-time progress updates."""
    
    # Drop missing values created by rolling window calculations
    df.dropna(inplace=True)

    # Select relevant features
    features = [
        "EMA_slow", "EMA_fast", "RSI", "BB_upper", "BB_middle", "BB_lower",
        "ATR", "MACD", "MACD_signal", "MACD_hist", "ROC", "ADX", 
        "DC_upper", "DC_lower", "DC_middle", "Volume_SMA", "Volatility"
    ]
    
    target = "signals"
    
    # Drop rows with missing values in the selected features or target
    df = df[features + [target]].dropna()
    
    print(f"Initial dataset size after dropping NaNs: {len(df)} rows")

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])

    # Convert into time-series sequences (shape: [samples, timesteps, features])
    X, y = [], []
    total_sequences = len(df) - window_size
    
    print(f"Processing time-series sequences ({total_sequences} samples)...")

    # **Create a TensorFlow-like progress bar**
    progbar = tf.keras.utils.Progbar(total_sequences)

    for i in range(total_sequences):
        X.append(df[features].iloc[i:i+window_size].values)  # Last N timesteps
        y.append(df[target].iloc[i+window_size])  # Target at t+window_size
        
        # Update progress bar every 500 steps
        if (i + 1) % 500 == 0 or (i + 1) == total_sequences:
            progbar.update(i + 1)

    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    """Create an LSTM model for trend classification."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),  # First LSTM layer
        Dropout(0.2),  # Regularization
        LSTM(32, return_sequences=False),  #Second LSTM layer
        Dropout(0.2),
        Dense(16, activation="relu"),  # Fully connected layer
        Dense(1, activation="sigmoid")  # Output layer (0 or 1)
    ])
    
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def build_optimized_lstm(input_shape, learning_rate, lstm_units, dropout_rate):
    """Builds a deeper optimized LSTM model."""
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=True),  # Added another LSTM layer
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=64, activation="relu"),  # Intermediate dense layer
        Dense(units=1, activation="sigmoid")  # Output layer
    ])
    
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
    return model



# Build the model



###########################################################################################################################################
# Load the data
dfmain = DataLoader.load_data()
df = dfmain.copy()
print (len(df))
window_size = 60  # Number of timesteps to look back
features = [
    "EMA_slow", "EMA_fast", "RSI", "BB_upper", "BB_middle", "BB_lower",
    "ATR", "MACD", "MACD_signal", "MACD_hist", "ROC", "ADX", 
    "DC_upper", "DC_lower", "DC_middle", "Volume_SMA", "Volatility"
]
input_shape = (window_size, len(features))


# generate reference signals
df = generate_perfect_signals(dfmain,20)

#Data Processing

X, y, scaler = preprocess_data(df, window_size)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

'''
print("Building LSTM Model...")
model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Save the model for future use

model_path = models_dir +"/LSTM_model.keras"
model.save(str(model_path))
print("Saved LSTM Model")
'''

#Hyper Tuning
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
print("Starting Hyperparameter tuning of LSTM Model...")
# Try different parameters
optimized_model = build_optimized_lstm(input_shape, learning_rate=0.0005, lstm_units=128, dropout_rate=0.2)

# Train the new model
history_opt = optimized_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

model_path = models_dir +"/Hypertuuned_LSTM_model.keras"
optimized_model.save(str(model_path))

print("Saved Optimized LSTM Model")

# Make predictions
y_pred_prob = optimized_model.predict(X_test)
# Convert predictions into NumPy array (if not already)
y_pred_prob = np.array(y_pred_prob)  # Ensure it's an array

# Apply thresholding
y_pred = (y_pred_prob > 0.5).astype(int)  


# Compare actual vs predicted
df_test = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})

# Save the comparison as a CSV file
df_test.to_csv("notebooks/actual_vs_predicted.csv", index=False)
print("Saved actual vs predicted labels to actual_vs_predicted.csv")

# Print the first 200 rows for verification
print(df_test.head(200))