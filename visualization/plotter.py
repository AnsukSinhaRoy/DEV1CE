#plotter.py
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot(df):
        fig, ax = plt.subplots(2, 1, figsize=(24, 20), sharex=True)

        ax[0].plot(df.index, df['portfolio_value'], label='Portfolio Value', color='blue')
        ax[0].set_title('Portfolio Value Over Time')
        ax[0].legend()
        
        ax[1].plot(df.index, df['close'], label='Close Price', color='black')
        ax[1].plot(df.index, df['EMA_fast'], label='Fast EMA (30)', color='green', linestyle='--')
        ax[1].plot(df.index, df['EMA_slow'], label='Slow EMA (75)', color='red', linestyle='--')
        ax[1].set_title('Price and EMAs')
        ax[1].legend()
        
        plt.show()
