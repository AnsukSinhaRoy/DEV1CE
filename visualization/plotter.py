import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot(df):
        fig, ax = plt.subplots(figsize=(24, 10))

        ax.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
        # Plot EMAs
        
        ax.plot(df.index, df['EMA_fast'], label='Fast EMA (30)', color='green', linestyle='--')
        ax.plot(df.index, df['EMA_slow'], label='Slow EMA (75)', color='red', linestyle='--')

        # Plot Donchian Channel levels
        ax.plot(df.index, df['DC_upper'], label='DC Upper (96)', color='orange', linestyle='--', alpha=0.7)
        ax.plot(df.index, df['DC_lower'], label='DC Lower (96)', color='purple', linestyle='--', alpha=0.7)
        ax.plot(df.index, df['DC_middle'], label='DC Middle (96)', color='gray', linestyle='--', alpha=0.7)

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
        #ax.plot(df.index, df['stop_loss'], label='Stop Loss', color='red', alpha=0.7)
        
        #ax.set_ylim(top=150000)

        ax.set_title('Trading Signals with Donchian Channel and Price')
        ax.legend()
        plt.show()