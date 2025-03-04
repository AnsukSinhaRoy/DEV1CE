class ema_strategy_long_only_stoploss_ATR:
    @staticmethod
    def run_strategy(row, prev_values, atr_factor=2):
        """Process a single row and return only the signal, incorporating ATR-based stop-loss."""

        close_price = row["close"]
        ema_slow = row["EMA_slow"]
        ema_fast = row["EMA_fast"]
        atr = row["ATR"]  # ATR for stop-loss calculation

        entered = prev_values["entered"]
        stop_loss = prev_values.get("stop_loss")  # Ensure stop_loss is not None
        signal = 0  # Default: No trade

        # Long Exit (Stop-Loss Hit)
        if entered and close_price <= stop_loss:
            signal = -1

        # Long Exit (EMA Condition)
        elif entered and ema_slow > close_price:
            signal = -1

        # Long Entry
        elif ema_fast > ema_slow and not entered:
            signal = 1

        return signal, close_price - (atr_factor * atr) if signal == 1 else stop_loss  # Return updated stop-loss
