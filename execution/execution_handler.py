class ExecutionHandler:
    def execute_order(self, symbol, quantity, price, order_type='market'):
        # Simulate order execution
        print(f"Executing {order_type} order for {quantity} shares of {symbol} at {price}")
        return True
