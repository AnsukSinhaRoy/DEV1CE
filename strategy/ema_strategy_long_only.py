class EMACrossoverStrategy_long:
    @staticmethod
    def run_strategy(row, prev_values):
        """Process a single row and return only the signal."""
        
        close_price = row["close"]
        ema_slow = row["EMA_slow"]
        ema_fast = row["EMA_fast"]

        entered = prev_values["entered"]
        signal = 0  # Default: No trade

         # Long Exit
        if entered and ema_slow > close_price:
            signal = -1

        # Long Entry
        elif ema_fast > ema_slow and not entered:
            signal = 1

        return signal,0
