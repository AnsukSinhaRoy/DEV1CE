# data_loader.py
import pandas as pd
from config import DATA_PATH

class DataLoader:
    @staticmethod
    def load_data():
        # Load the data and parse the 'datetime' column
        df = pd.read_csv(DATA_PATH, parse_dates=['datetime'])

        # Ensure the 'datetime' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        # Sort by datetime
        df.sort_values('datetime', inplace=True)

        # Extract the date from the datetime column
        df['date'] = df['datetime'].dt.date

        return df