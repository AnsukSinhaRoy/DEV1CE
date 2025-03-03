
class EMACrossoverStrategy_long_short:
    @staticmethod
    def run_strategy(row, prev_values):
        """Process a single row and return updated strategy state."""
        
        close_price = row["close"]
        ema_slow = row["EMA_slow"]
        ema_fast = row["EMA_fast"]
        
        # Retrieve previous state
        entered = prev_values["entered"]
        short_entered = prev_values["short_entered"]
        shares = prev_values["shares"]
        amount = prev_values["amount"]
        signal = 0  # Default: No trade

        # Initialize investment amount if not set
        if amount is None:
            amount = close_price

        # LONG SELL
        if entered and (ema_slow - close_price) > 0:
            signal = -1  # Long exit
            amount = shares * close_price
            shares = 0
            entered = False
            #short_entered = True  # Prepare to enter a short trade

        # LONG BUY
        elif (ema_fast - ema_slow) > 0 and not short_entered and not entered:
            signal = 1  # Long entry
            entered = True
            shares = amount / close_price
            short_entered = False  # Reset short_entered after entering

        elif not short_entered and not entered and (ema_slow - ema_fast) > 0:
            signal = -2
            short_entered = True
            shares = amount / close_price
            entered = False

        elif short_entered and (close_price - ema_slow) > 0:
            signal = 2
            amount = shares * close_price
            shares = 0
            short_entered = False
        # Calculate portfolio value
        portfolio_value = amount if not entered else shares * close_price

        return {
            "entered": entered,
            "short_entered": short_entered,
            "shares": shares,
            "amount": amount,
            "signal": signal,
            "portfolio_value": portfolio_value,
        }
