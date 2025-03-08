import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
models_dir = parent_dir +"/models"

# Import custom modules
from data.data_loader import DataLoader
from visualization.plotter import Plotter

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom DataLoader (Assumed)
dfmain = DataLoader.load_data()
df = dfmain.copy()
print (len(df))

# Define features
features = [
    "EMA_slow", "EMA_fast", "RSI", "BB_upper", "BB_middle", "BB_lower",
    "ATR", "MACD", "MACD_signal", "MACD_hist", "ROC", "ADX", 
    "DC_upper", "DC_lower", "DC_middle", "Volume_SMA", "Volatility"
]
target = "signals"

# Signal generation function
def generate_perfect_signals(df, lookahead):
    df = df.copy()
    df["signals"] = np.nan
    for i in range(len(df) - lookahead):
        future_prices = df["close"].iloc[i + 1 : i + 1 + lookahead].values
        if len(future_prices) > 0:
            max_future_price = np.max(future_prices)
            min_future_price = np.min(future_prices)
            if df["close"].iloc[i] <= min_future_price + 0.1:
                df.at[i, "signals"] = 1  # Uptrend
            elif df["close"].iloc[i] >= max_future_price - 0.1:
                df.at[i, "signals"] = 0  # Downtrend
    df["signals"] = df["signals"].ffill().fillna(0)
    return df

# Process dataset
df = generate_perfect_signals(df, lookahead=20)
df.dropna(inplace=True)

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
df[features] = scaler.fit_transform(df[features])

# Convert to sequences
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length
sequence_length = 60  
X, y = create_sequences(df[features].values, df[target].values, sequence_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# Create DataLoaders for mini-batch training
batch_size = 64
train_dataset = TensorDataset(X_train_torch, y_train_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take only last output
        return self.sigmoid(out)

# Instantiate model
input_size = len(features)
model = LSTMModel(input_size=input_size).to(device)

# Loss function & optimizer
criterion = nn.BCELoss()  # Binary cross-entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
epochs = 100
early_stop_patience = 10
best_loss = float("inf")
patience_counter = 0

print("Training LSTM model on GPU..." if torch.cuda.is_available() else "Training on CPU...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "optimized_lstm.pth")  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load("optimized_lstm.pth"))

# Make predictions
model.eval()
y_pred_prob = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        y_pred_prob.extend(outputs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Convert predictions
y_pred_prob = np.array(y_pred_prob).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

# Save predictions to CSV
df_test = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
df_test.to_csv("actual_vs_predicted_pytorch.csv", index=False)
print("Saved actual vs predicted labels to actual_vs_predicted_pytorch.csv")

# Print sample predictions
print(df_test.head(200))
