import pandas as pd
import os

class Backtester:
    @staticmethod
    def run_backtest(df):
        final_value = df["portfolio_value"].iloc[-1]
        initial_value = df["close"].iloc[0]
        return_percentage = (final_value - initial_value) / initial_value * 100
        print(f"Percentage Return: {return_percentage:.2f}%")

        df.to_csv('output/FullData.csv', index=False)
        df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'signals']].to_csv('output/signals.csv', index=False)

        # Another copy of signals for untrade backtesting
        output_dir = "../../TradingAlgorithms/My_Algorithms"
        signals_path = os.path.join(output_dir, "signals.csv")
        df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'signals']].to_csv(signals_path, index=False)
        
