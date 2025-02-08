import pandas as pd
import numpy as np

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window=30, long_window=60):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        data['short_mavg'] = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        data['long_mavg'] = data['close'].rolling(window=self.long_window, min_periods=1).mean()
        data['signal'] = 0
        data['signal'][self.short_window:] = \
            np.where(data['short_mavg'][self.short_window:] > data['long_mavg'][self.short_window:], 1, 0)
        data['positions'] = data['signal'].diff()
        return data

