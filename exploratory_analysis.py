import matplotlib.pyplot as plt
import logging

def calculate_moving_averages(df):
    logging.info("Calculating moving averages...")
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

def calculate_volatility(df):
    logging.info("Calculating volatility...")
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df

def plot_data(df, ticker_symbol):
    logging.info("Plotting data...")
    # Plot Closing Price with Moving Averages
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Closing Price')
    plt.plot(df['MA20'], label='20-Day MA')
    plt.plot(df['MA50'], label='50-Day MA')
    plt.title(f'{ticker_symbol} Closing Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.savefig(f'{ticker_symbol}_Moving_Averages.png')
    plt.close()
    logging.info(f'Plot saved as {ticker_symbol}_Moving_Averages.png')

    # Plot Volatility
    plt.figure(figsize=(14, 7))
    plt.plot(df['Volatility'], label='20-Day Volatility', color='orange')
    plt.title(f'{ticker_symbol} 20-Day Volatility')
    plt.xlabel('Date')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.savefig(f'{ticker_symbol}_Volatility.png')
    plt.close()
    logging.info(f'Plot saved as {ticker_symbol}_Volatility.png')
