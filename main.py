import pandas as pd
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from portfolio.portfolio import Portfolio
from execution.execution_handler import ExecutionHandler
from backtesting.backtest import Backtest

if __name__ == "__main__":
    # Load historical data
    data = pd.read_csv('data/historical_data.csv', parse_dates=True, index_col='datetime')

    # Initialize components
    strategy = MovingAverageCrossoverStrategy()
    portfolio = Portfolio()
    execution_handler = ExecutionHandler()

    # Run backtest
    backtest = Backtest(data, strategy, portfolio, execution_handler)
    portfolio_values = backtest.run()

    # Output results
    print(f"Final Portfolio Value: {portfolio_values[-1]}")
