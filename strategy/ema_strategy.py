from config import TRAILING_STOP_PCT

class EMACrossoverStrategy:
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
        stop_loss = prev_values["stop_loss"]
        signal = 0  # Default: No trade

        # Initialize investment amount if not set
        if amount is None:
            amount = close_price

        # Update trailing stop-loss if in a long position
        if entered and stop_loss is not None:
            stop_loss = max(stop_loss, close_price * (1 - TRAILING_STOP_PCT))

        # Long Exit
        if entered and (ema_slow - close_price) > 0:
            signal = -1
            amount = shares * close_price
            shares = 0
            entered = False
            stop_loss = None

        # Long Entry
        elif (ema_slow - ema_fast) < 0 and not entered:
            signal = 1
            shares = amount / close_price
            entered = True
            stop_loss = close_price * (1 - TRAILING_STOP_PCT)  # Set initial stop-loss

        # Calculate portfolio value
        portfolio_value = amount if not entered else shares * close_price

        return {
            "entered": entered,
            "shares": shares,
            "amount": amount,
            "stop_loss": stop_loss,
            "signal": signal,
            "portfolio_value": portfolio_value
        }
