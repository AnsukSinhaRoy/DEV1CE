import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot(df):
        fig, ax = plt.subplots(figsize=(24, 10))

        # Plot price and EMAs
        ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
        ax.plot(df.index, df['EMA_fast'], label='Fast EMA (30)', color='green', linestyle='--')
        ax.plot(df.index, df['EMA_slow'], label='Slow EMA (75)', color='red', linestyle='--')

        # Plot signals
        buy_signals = df[df["signals"] == 1]
        sell_signals = df[df["signals"] == -1]
        short_entry_signals = df[df["signals"] == 2]
        short_exit_signals = df[df["signals"] == -2]

        ax.scatter(buy_signals.index, buy_signals["close"], color="green", marker="^", label="Buy Signal", s=100)
        ax.scatter(sell_signals.index, sell_signals["close"], color="red", marker="v", label="Sell Signal", s=100)
        ax.scatter(short_entry_signals.index, short_entry_signals["close"], color="blue", marker="v", label="Short Entry", s=100)
        ax.scatter(short_exit_signals.index, short_exit_signals["close"], color="purple", marker="^", label="Short Exit", s=100)
        ax.plot(df.index, df['portfolio_value'], label='Portfolio Value', color='blue', alpha=0.7)
        ax.plot(df.index, df['close']-(2*df['ATR']), label='calculated ATR', color='red', alpha=0.7)

        ax.set_title('Trading Signals with EMAs and Price')
        ax.legend()
        plt.show()
