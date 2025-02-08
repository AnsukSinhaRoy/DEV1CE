# Modular Trading Algorithm

This project implements a **modular trading algorithm** using **Object-Oriented Programming (OOP)** principles. It includes data loading, indicator calculations, backtesting, and visualization.

## 📁 Project Structure
```plaintext
trading_algorithm/
│── data/
│   ├── data_loader.py        # Loads and processes data from CSV
│   ├── indicators.py         # Calculates technical indicators using TA-Lib
│
│── strategy/
│   ├── base_strategy.py      # Base class for trading strategies
│   ├── ema_strategy.py       # Implements EMA crossover strategy
│
│── execution/
│   ├── backtester.py         # Runs backtests on historical data
│
│── visualization/
│   ├── plotter.py            # Handles visualization of trading results
│
│── main.py                   # Entry point for the program
│── config.py                 # Configuration file for paths and parameters
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation



## 📌 Features
- **Modular Design** using OOP.
- **Dynamic Technical Indicators** (EMA, RSI, Bollinger Bands, MACD, ATR, Stochastic Oscillator).
- **Backtesting Framework** to evaluate strategy performance.
- **Data Visualization** for better insights.

## 🛠️ Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/AnsukSinhaRoy/DEV1CE.git
    cd trading_algorithm
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the trading algorithm**:
    ```bash
    python main.py
    ```

## 📊 Strategies Implemented
- **EMA Crossover Strategy**: Buys when the **fast EMA (30-period) crosses above the slow EMA (75-period)** and sells when it crosses below.

## 🏆 Backtesting
- The **Backtester** calculates portfolio value and saves results in:
    - `output/FullData.csv`
    - `output/signals.csv`
- **Performance Metrics**:
    - Initial and Final Portfolio Value
    - Percentage Return

## 📈 Visualization
- The **Plotter** module generates:
    - **Portfolio Value Over Time**
    - **Price & Moving Averages (EMA 30 & 75)**

## 📢 Contributing
- Fork the repo 🍴
- Create a feature branch 🌿
- Make your changes ✨
- Submit a pull request 📩


🚀 Happy Trading! 
