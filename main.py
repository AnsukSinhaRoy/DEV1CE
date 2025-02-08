#main.py
from data.data_loader import DataLoader
from data.indicators import Indicators
from strategy.ema_strategy import EMACrossoverStrategy
from execution.backtester import Backtester
from visualization.plotter import Plotter

def main():
    df = DataLoader.load_data()
    df = Indicators.add_indicators(df)
    strategy = EMACrossoverStrategy(df)
    Backtester.run_backtest(strategy)
    Plotter.plot(df)

if __name__ == "__main__":
    main()
