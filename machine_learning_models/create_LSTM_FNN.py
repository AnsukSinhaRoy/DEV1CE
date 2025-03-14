import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
parent_dir = r"E:\Python\GIT - DEV1CE\DEV1CE"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data.data_loader import DataLoader as dl
from strategy.perfect_signal import generate_perfect_signals
from machine_learning_models.LSTM_FNN.data_processing import preprocess_data
from config import SLOW_EMA_PERIOD

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # For progress bar

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dfmain = dl.load_data()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"🔥 Using device: {device}")

# Input features array for LSTM model
LSTMfeatures = [
    # Core OHLCV Data
    #"open", "high", "low", 

    #Volume
    #"volume","volume_change", "Volume_SMA", "Volume_Threshold", "OBV", "CMF",
    
    # Trend Indicators
    "EMA_slow", "EMA_fast", "RSI", "MACD", "ATR",
    #"Ichimoku_Tenkan", "Ichimoku_Kijun", "Parabolic_SAR",

    # Volatility Indicators
    #"ATR", "Norm_ATR", "BB_upper", "BB_middle", "BB_lower", "Volatility",

    # Momentum Indicators
    #"ROC", "slowk", "slowd", "ADX",

    # Time Features
    "seconds", "close", # Add more time features if available
]

# Output features for prediction
LSTM_Output_features = ["EMA_slow", "EMA_fast",]
window_size = 30
future_steps = 1

# Define LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size=len(LSTMfeatures), hidden_dim=128, num_layers=8, output_size=len(LSTM_Output_features) * future_steps, dropout_prob=0.5):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last timestep output
        x = self.fc(x)
        x = x.view(x.size(0), future_steps, len(LSTM_Output_features))
        return x

df = dfmain[:67000]
df = generate_perfect_signals(df, lookahead=30)

# Preprocess the data
X, y, processed_df, scaler = preprocess_data(LSTMfeatures, df, window_size,LSTM_Output_features, future_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Model Initialization
print("\n🚀 Instantiating LSTM model...")
input_size = X_train.shape[2]
output_size = len(LSTM_Output_features) * future_steps
lstm_model = LSTMModel(input_size=input_size, output_size=output_size).to(device)
print(f"✅ Model created with input_size={input_size}, output_size={output_size}")
print(f"✅ Device: {device}")

# Loss & Optimizer
print("\n⚙️ Configuring training components...")
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.0005839081038527126)
print(f"✅ Loss: {criterion.__class__.__name__}")
print(f"✅ Optimizer: {optimizer.__class__.__name__} (lr={0.0005839081038527126:.6f})")

# Data Preparation
print("\n🛠️ Converting data to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
print(f"✅ Training data shape: {X_train_tensor.shape} (X), {y_train_tensor.shape} (y)")
print(f"✅ Test data shape: {X_test_tensor.shape} (X), {y_test_tensor.shape} (y)")

# DataLoader Setup
print("\n📦 Creating DataLoader...")
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Batch size: {batch_size}")
print(f"✅ Total batches per epoch: {len(train_loader)}")

# Training Loop
print("\n🎯 Starting training...")
epochs = 10
for epoch in range(epochs):
    print(f"\n-----------------------------")
    print(f"🌟 Epoch {epoch+1}/{epochs}")
    print(f"-----------------------------")
    
    lstm_model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")

    for batch_X, batch_y in progress_bar:
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        batch_y = batch_y.view(-1, future_steps, len(LSTM_Output_features))
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

# Model Evaluation
lstm_model.eval()
with torch.no_grad():
    y_pred_tensor = lstm_model(X_test_tensor)
y_pred_numpy = y_pred_tensor.cpu().numpy()
y_test_flat = y_test.reshape(-1, y_test.shape[-1])
y_pred_flat = y_pred_numpy.reshape(-1, y_pred_numpy.shape[-1])

# Metrics
mse = mean_squared_error(y_test_flat, y_pred_flat)
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)
print(f"\n💡 Test MSE: {mse:.6f}")
print(f"💡 Test MAE: {mae:.6f}")
print(f"💡 R² Score: {r2:.4f}")

# 🔹 Plot Actual vs Predicted (First 100 Samples)
plt.figure(figsize=(10, 5))
plt.plot(y_test_flat[:], label="Actual", linestyle="dashed", color="blue")
plt.plot(y_pred_flat[:], label="Predicted", alpha=0.8, color="red")
plt.legend()
plt.title("LSTM Predictions vs Actual Values on Test Data")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price (or Target Variable)")
plt.show()