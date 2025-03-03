#ema_strategy.py
class EMACrossoverStrategy_long:
    @staticmethod
    def run_strategy(row, prev_values):
        """Process a single row and return updated strategy state."""
        
        close_price = row["close"]
        ema_slow = row["EMA_slow"]
        ema_fast = row["EMA_fast"]
        
        # Retrieve previous state
        entered = prev_values["entered"]
        shares = prev_values["shares"]
        amount = prev_values["amount"]
        signal = 0  # Default: No trade

        # Initialize investment amount if not set
        if amount is None:
            amount = close_price

        # Long Exit
        if entered and (ema_slow - close_price) > 0:
            signal = -1
            amount = shares * close_price
            shares = 0
            entered = False

        # Long Entry
        elif (ema_slow - ema_fast) < 0:
            signal = 1
            shares = amount / close_price
            entered = True

        # Calculate portfolio value
        portfolio_value = amount if not entered else shares * close_price

        return {
            "short_entered": False,
            "entered": entered,
            "shares": shares,
            "amount": amount,
            "signal": signal,
            "portfolio_value": portfolio_value
        }
