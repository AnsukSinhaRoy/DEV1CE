#sequential_indicators
class SequentialIndicators:
    def __init__(self, slow_ema=50, fast_ema=20):
        self.slow_ema_period = slow_ema
        self.fast_ema_period = fast_ema
        
        # EMA tracking
        self.slow_ema = None
        self.fast_ema = None
        self.prev_slow_ema = None
        self.prev_fast_ema = None
        self.slow_alpha = 2 / (slow_ema + 1)
        self.fast_alpha = 2 / (fast_ema + 1)

    def update_indicators(self, row, idx):
        # EMA Calculation
        if idx == 0:
            self.slow_ema = row['close']
            self.fast_ema = row['close']
        else:
            self.slow_ema = row['close'] * self.slow_alpha + \
                self.prev_slow_ema * (1 - self.slow_alpha)
            self.fast_ema = row['close'] * self.fast_alpha + \
                self.prev_fast_ema * (1 - self.fast_alpha)
        
        row['EMA_slow'] = self.slow_ema
        row['EMA_fast'] = self.fast_ema
        
        self.prev_slow_ema = self.slow_ema
        self.prev_fast_ema = self.fast_ema
        
        return row