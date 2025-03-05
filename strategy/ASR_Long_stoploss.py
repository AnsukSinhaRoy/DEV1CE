def run_strategy_ASR_Long_stoploss(row, prev_values, atr_factor=2):
    """Process a single row and return the signal with a dynamic ATR-based stop-loss."""

    close_price = row["close"]
    ema_slow = row["EMA_slow"]
    ema_fast = row["EMA_fast"]
    atr = row["ATR"]  

    entered = prev_values["entered"]
    stop_loss = prev_values.get("stop_loss") or 0  # Default to 0 if None
    signal = 0  # Default: No trade

    # **Long Exit (Stop-Loss Hit)**
    if entered and close_price <= stop_loss:
        signal = -1

    # **Long Exit (EMA Condition)**
    elif entered and ema_slow > close_price:
        signal = -1

    # **Long Entry (With Initial Stop-Loss)**
    elif ema_fast > ema_slow and not entered and close_price > ema_fast:
        signal = 1
        stop_loss = close_price - (atr_factor * atr)  # Set initial stop-loss

    # **Trailing Stop-Loss (For Active Trades)**
    elif entered:
        new_stop_loss = close_price - (atr_factor * atr)  
        stop_loss = max(stop_loss, new_stop_loss)  # Never increase stop-loss for long trades

    return signal, stop_loss  # Return updated stop-loss
