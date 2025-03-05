from data.data_loader import DataLoader
import data.indicators as Indicators
from execution.backtester import Backtester
from visualization.plotter import Plotter

#import strategies
from strategy.ema_strategy_long_only import EMACrossoverStrategy_long
from strategy.ema_stratagy_long_short import EMACrossoverStrategy_long_short
from strategy.ASR_Long import run_strategy_ASR_Long
from strategy.ASR_Long_stoploss import run_strategy_ASR_Long_stoploss
from strategy.LWVolatilityBreakoutStrategy import DonchianMomentumStrategy


def main():
    # Load and preprocess data
    dfmain = DataLoader.load_data()
    df = dfmain.iloc[0:100000].copy()
    df = Indicators.add_indicators(df)
    
    prev_values = initialize_state_variables()

    # Process each row
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # STEP 1:  Get signal from different strategies
        signal_EMA_long = EMACrossoverStrategy_long.run_strategy_EMA_Long(df, i, prev_values, window=3)
        signal_long_short = EMACrossoverStrategy_long_short.run_strategy(row, prev_values)
        signal_ASR_long = run_strategy_ASR_Long(row, prev_values)
        signal_ASR_long_stoploss = run_strategy_ASR_Long_stoploss(row, prev_values)
        lwvolatility = DonchianMomentumStrategy.run_strategy(df, i, prev_values)
        
        chosen_strategy = lwvolatility


        signal = chosen_strategy[0]
        # STEP 2: Update portfolio according to the signal
        prev_values["stop_loss"] = chosen_strategy[1]
        prev_values = update_portfolio(signal, row, prev_values)


        # STEP 3: Update the dataframe with the chosen signal and portfolio value
        df.loc[i, "signals"] = signal
        df.loc[i, "portfolio_value"] = prev_values["portfolio_value"]
        df.loc[i, "shares"] = prev_values["shares"]
        df.loc[i,"stop_loss"] = chosen_strategy[1]

    Backtester.run_backtest(df)
    Plotter.plot(df)

def initialize_state_variables():
    return {
        "short_entered": False,
        "entered": False,
        "shares": 0,
        "amount": 10000.0,
        "portfolio_value": 10000.0,
        "stop_loss": None,
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
            amount = shares * close_price
            shares = 0
            entered = False

    elif signal == 2:  # Short Entry
        if not entered and not short_entered:
            shares = amount / close_price
            amount = 0
            entered = False
            short_entered = True

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
