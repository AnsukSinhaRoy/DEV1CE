#ema_strategy.py 
from strategy.base_strategy import BaseStrategy
from config import TRAILING_STOP_PCT

class EMACrossoverStrategy(BaseStrategy):
    def run_strategy(self):#self.df is the dataframe that will be passed to the strategy
        self.df["signals"] = 0

        for i in range(1, len(self.df)):
            if self.entered:
                self.stop_loss = max(self.stop_loss, self.df["close"].iloc[i] * (1 - TRAILING_STOP_PCT))

            if self.entered and ((self.df["EMA_slow"].iloc[i] - self.df["close"].iloc[i]) > 0):
                self.df.loc[i, "signals"] = -1
                self.amount = self.shares * self.df["close"].iloc[i]
                self.shares = 0
                self.entered = False
                self.stop_loss = None

            elif ((self.df["EMA_slow"].iloc[i] - self.df["EMA_fast"].iloc[i]) < 0 and not self.entered):
                self.df.loc[i, "signals"] = 1
                self.shares = self.amount / self.df["close"].iloc[i]
                self.entered = True
                self.stop_loss = self.df["close"].iloc[i] * (1 - TRAILING_STOP_PCT)

            self.df.loc[i, "portfolio_value"] = self.amount if not self.entered else self.shares * self.df["close"].iloc[i]
            self.df.loc[i, "shares"] = self.shares

        return self.df
