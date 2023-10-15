import yfinance as yf
import pandas as pd
import logging

def fetch_stock_data(ticker_symbol, start_date='2010-01-01', end_date='2010-10-31'):
    logging.info(f"Fetching stock data for {ticker_symbol} from {start_date} to {end_date}...")
    ticker_data = yf.Ticker(ticker_symbol)
    df = ticker_data.history(period='1d', start=start_date, end=end_date)
    if not df.empty:
        logging.info("Data retrieval successful.")
    else:
        logging.warning("Data retrieval failed or returned empty dataset.")
    return df
