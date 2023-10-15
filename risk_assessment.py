import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def calculate_daily_returns(df):
    logging.info("Calculating daily returns...")
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def plot_return_distribution(df, ticker_symbol):
    logging.info("Plotting return distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Daily_Return'].dropna(), bins=50, kde=True)
    plt.title(f'{ticker_symbol} Daily Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig(f'{ticker_symbol}_Return_Distribution.png')
    plt.close()
    logging.info(f'Return distribution plot saved as {ticker_symbol}_Return_Distribution.png')

def calculate_risk_metrics(df):
    logging.info("Calculating risk metrics...")
    volatility = df['Daily_Return'].std() * np.sqrt(252)
    annualized_return = df['Daily_Return'].mean() * 252
    sharpe_ratio = (annualized_return - 0.02) / volatility
    return volatility, annualized_return, sharpe_ratio
