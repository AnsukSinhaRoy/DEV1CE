#backtester.py
import pandas as pd

class Backtester:
    @staticmethod
    def run_backtest(strategy):
        df = strategy.apply_strategy()
        final_value = df["portfolio_value"].iloc[-1]
        initial_value = df["close"].iloc[0]
        return_percentage = (final_value - initial_value) / initial_value * 100
        print(f"Percentage Return: {return_percentage:.2f}%")
        df.to_csv('output/FullData.csv', index=False)
        df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'signals']].to_csv('output/signals.csv', index=False)
