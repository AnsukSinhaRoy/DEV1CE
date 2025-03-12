import numpy as np
import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
parent_dir = r"E:\Python\GIT - DEV1CE\DEV1CE"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data.data_loader import DataLoader
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

dfmain = DataLoader.load_data()
print(dfmain.head())    
