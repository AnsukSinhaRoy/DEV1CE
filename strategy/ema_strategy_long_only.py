class EMACrossoverStrategy_long:
    @staticmethod
    def run_strategy_EMA_Long(df, i, prev_values, window=3, threshold=7.5):
        """
        Process a single row and return the signal.
        Uses a rolling window to detect actual crossovers and avoids sideways markets.
        """
        if i < window:  # Not enough data for a proper crossover check
            return 0
        
        close_price = df.iloc[i]["close"]
        ema_slow = df.iloc[i]["EMA_slow"]
        ema_fast = df.iloc[i]["EMA_fast"]

        # Get past values to detect crossover
        ema_fast_prev = df.iloc[i - window]["EMA_fast"]
        ema_slow_prev = df.iloc[i - window]["EMA_slow"]

        entered = prev_values["entered"]
        signal = 0  # Default: No trad

        # **Detect Long Entry (Fast EMA crosses above Slow EMA)**
        if ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow and not entered:
            signal = 1  # Long entry

        # **Detect Long Exit (Fast EMA crosses below Slow EMA)**
        elif entered and ema_fast < ema_slow:
            signal = -1  # Long exit

        return signal
