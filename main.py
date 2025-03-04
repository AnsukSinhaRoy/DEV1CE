from data.data_loader import DataLoader
import data.indicators as Indicators
from execution.backtester import Backtester
from visualization.plotter import Plotter

#import strategies
from strategy.ema_strategy_long_only import EMACrossoverStrategy_long
from strategy.ema_stratagy_long_short import EMACrossoverStrategy_long_short
from strategy.ema_strategy_long_only_stoploss_ATR import ema_strategy_long_only_stoploss_ATR


def main():
    # Load and preprocess data
    df = DataLoader.load_data()
    df = Indicators.add_indicators(df)
    
    prev_values = initialize_state_variables()

    # Process each row
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # STEP 1:  Get signal from different strategies
        signal_long = EMACrossoverStrategy_long.run_strategy(row, prev_values)
        signal_long_short = EMACrossoverStrategy_long_short.run_strategy(row, prev_values)
        signal_long_stoploss_ATR = ema_strategy_long_only_stoploss_ATR.run_strategy(row, prev_values)
        
        
        chosen_signal, stop_loss = signal_long_short

        # STEP 2: Update portfolio according to the signal
        prev_values["stop_loss"] = stop_loss
        prev_values = update_portfolio(chosen_signal, row, prev_values)


        # STEP 3: Update the dataframe with the chosen signal and portfolio value
        df.loc[i, "signals"] = chosen_signal
        df.loc[i, "portfolio_value"] = prev_values["portfolio_value"]
        df.loc[i, "shares"] = prev_values["shares"]

    Backtester.run_backtest(df)
    Plotter.plot(df)

def initialize_state_variables():
    return {
        "short_entered": False,
        "entered": False,
        "shares": 0,
        "amount": 10000.0,  # Initial cash balance
        "portfolio_value": 10000.0,
        "stop_loss": None,  # Store stop-loss value
    }


def update_portfolio(signal, row, prev_values):
    close_price = row["close"]
    shares = prev_values["shares"]
    amount = prev_values["amount"]
    entered = prev_values["entered"]
    short_entered = prev_values["short_entered"]
    stop_loss = prev_values["stop_loss"]

    if signal == 1:  # Long Entry
        if not entered and not short_entered:
            shares = amount / close_price
            amount = 0  # Invest all cash
            entered = True
            short_entered = False

    elif signal == -1:  # Long Exit
        if entered:
            amount = shares * close_price  # Convert shares to cash
            shares = 0
            entered = False

    elif signal == 2:  # Short Entry
        if not entered and not short_entered:
            shares = amount / close_price
            amount = 0
            short_entered = True
            entered = False

    elif signal == -2:  # Short Exit
        if short_entered:
            amount = shares * close_price
            shares = 0
            short_entered = False

    # Update portfolio value
    portfolio_value = amount + (shares * close_price)

    return {
        "short_entered": short_entered,
        "entered": entered,
        "shares": shares,
        "amount": amount,
        "portfolio_value": portfolio_value,
        "stop_loss": stop_loss,
    }

def update_portfolio_stoploss_ATR(signal, row, prev_values, atr_factor=2):
    close_price = row["close"]
    atr = row["ATR"]  # Use ATR for stop-loss calculation
    shares = prev_values["shares"]
    amount = prev_values["amount"]
    entered = prev_values["entered"]
    short_entered = prev_values["short_entered"]
    stop_loss = prev_values["stop_loss"]

    if signal == 1:  # Long Entry
        if not entered and not short_entered:
            shares = amount / close_price
            amount = 0  # Invest all cash
            entered = True
            short_entered = False
            stop_loss = close_price - (atr_factor * atr)  # ATR-based stop-loss below entry

    elif signal == -1 or (entered and close_price <= stop_loss):  # Long Exit (Signal or Stop-Loss Hit)
        if entered:
            amount = shares * close_price  # Convert shares to cash
            shares = 0
            entered = False
            stop_loss = None  # Reset stop-loss

    elif signal == -2:  # Short Entry
        if not entered and not short_entered:
            shares = amount / close_price
            amount = 0
            short_entered = True
            entered = False
            stop_loss = close_price + (atr_factor * atr)  # ATR-based stop-loss above entry

    elif signal == 2 or (short_entered and close_price >= stop_loss):  # Short Exit (Signal or Stop-Loss Hit)
        if short_entered:
            amount = shares * close_price
            shares = 0
            short_entered = False
            stop_loss = None  # Reset stop-loss

    # Update portfolio value
    portfolio_value = amount + (shares * close_price)

    return {
        "short_entered": short_entered,
        "entered": entered,
        "shares": shares,
        "amount": amount,
        "portfolio_value": portfolio_value,
        "stop_loss": stop_loss,  # Store updated stop-loss
    }



if __name__ == "__main__":
    main()
