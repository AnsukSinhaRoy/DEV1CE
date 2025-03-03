#main.py
from data.data_loader import DataLoader
from data.indicators import Indicators
from execution.backtester import Backtester
from visualization.plotter import Plotter

#import strategies
from strategy.ema_strategy_long_only import EMACrossoverStrategy_long
from strategy.ema_stratagy_long_short import EMACrossoverStrategy_long_short


def main():
    # Load and preprocess data
    df = DataLoader.load_data()
    df = Indicators.add_indicators(df)
    
    #Initialize signals, portfolio value and shares
    df = initialize_signals(df)
    
    prev_values = initialize_state_variables()

    # Process each row
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        EMA_Crossover_long_only = EMACrossoverStrategy_long.run_strategy(row, prev_values)
        EMA_Crossover_long_short = EMACrossoverStrategy_long_short.run_strategy(row, prev_values)

        chosen_signal = EMA_Crossover_long_short
        df.loc[i, "signals"] = chosen_signal["signal"]
        df.loc[i, "portfolio_value"] = chosen_signal["portfolio_value"]
        df.loc[i, "shares"] = chosen_signal["shares"]

        prev_values = chosen_signal

    Backtester.run_backtest(df)

    Plotter.plot(df)
def initialize_signals(df):
    df["signals"] = 0
    df["portfolio_value"] = 10000.0
    df["shares"] = 0.0
    return df

def initialize_state_variables():
    return {
        "short_entered": False,
        "entered": False,
        "shares": 0,
        "amount": None,
        "stop_loss": None
    }

if __name__ == "__main__":
    main()
