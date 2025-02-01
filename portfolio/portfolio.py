class Portfolio:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital
        self.positions = {}
        self.cash = initial_capital
        self.portfolio_value = []

    def update_portfolio(self, symbol, quantity, price):
        if symbol in self.positions:
            self.positions[symbol] += quantity
        else:
            self.positions[symbol] = quantity
        self.cash -= quantity * price

    def calculate_portfolio_value(self, market_data):
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            total_value += quantity * market_data[symbol]['close']
        self.portfolio_value.append(total_value)
        return total_value
