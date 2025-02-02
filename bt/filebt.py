# backtesting/backtest.py

class Backtest:
    def __init__(self, data, strategy, portfolio):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio

    def run(self):
        print(f"Running backtest with strategy: {self.strategy.__class__.__name__}")
        self.strategy.apply(self.data)
        print("Backtest completed!")

# Ensure this module can be imported properly
if __name__ == "__main__":
    print("Backtest module loaded successfully.")
