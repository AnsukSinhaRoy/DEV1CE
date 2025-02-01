import pandas as pd
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from portfolio.portfolio import Portfolio
from execution.execution_handler import ExecutionHandler

class Backtest:
    def __init__(self, data, strategy, portfolio, execution_handler):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler

    def run(self):
        signals = self.strategy.generate_signals(self.data)
        for index, row in signals.iterrows():
            if row['positions'] == 1:
                self.execution_handler.execute_order('AAPL', 100, row['close'])
                self.portfolio.update_portfolio('AAPL', 100, row['close'])
            elif row['positions'] == -1:
                self.execution_handler.execute_order('AAPL', -100, row['close'])
                self.portfolio.update_portfolio('AAPL', -100, row['close'])
            self.portfolio.calculate_portfolio_value({'AAPL': row})
        return self.portfolio.portfolio_value
