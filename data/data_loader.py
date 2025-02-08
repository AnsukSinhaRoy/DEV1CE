#data_loader.py
import pandas as pd
from config import DATA_PATH

class DataLoader:
    @staticmethod
    def load_data():
        df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])
        df.sort_values('datetime', inplace=True)
        df['date'] = df['datetime'].dt.date
        return df
