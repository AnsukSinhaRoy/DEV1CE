#base_strategy.py
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, df):
        self.df = df
        self.shares = 0
        self.amount = df['close'][0]
        self.entered = False
        self.stop_loss = None
    
    @abstractmethod
    def apply_strategy(self):
        pass
