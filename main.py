import os
import data_retrieval
import exploratory_analysis as eda
import predictive_models as pm
import backtesting as bt
import risk_assessment as ra
import utils
import matplotlib.pyplot as plt
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_insights(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    assistant_reply = response.choices[0].message.content
    return assistant_reply

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Specify the ticker symbol and date range
    ticker_symbol = 'AAPL'  # You can change this to any ticker symbol you like
    start_date = '2010-01-01'
    end_date = '2010-10-31'
    
    # Step 1: Data Retrieval
    logging.info("Step 1: Fetching Data...")
    df = data_retrieval.fetch_stock_data(ticker_symbol, start_date, end_date)
    
    # Check if data retrieval was successful
    if df.empty:
        logging.error("Failed to retrieve data. Please check the ticker symbol and date range.")
        return

    # Step 2: Exploratory Data Analysis
    logging.info("Step 2: Performing Exploratory Data Analysis...")
    df = eda.calculate_moving_averages(df)
    df = eda.calculate_volatility(df)
    eda.plot_data(df, ticker_symbol)
    
    # Step 3: Predictive Modeling
    logging.info("Step 3: Generating Predictions...")

    # ARIMA Model
    logging.info("ARIMA Model Prediction...")
    arima_forecast = pm.predict_with_arima(df)

    # Prophet Model
    logging.info("Prophet Model Prediction...")
    prophet_forecast = pm.predict_with_prophet(df)

    # LSTM Model
    logging.info("LSTM Model Prediction...")
    lstm_predictions, y_test = pm.predict_with_lstm(df)

    # SARIMA Model
    logging.info("SARIMA Model Prediction...")
    sarima_forecast = pm.predict_with_sarima(df)

    # GARCH Model
    logging.info("GARCH Model Prediction...")
    garch_forecast = pm.predict_with_garch(df)

    # Random Forest Model
    logging.info("Random Forest Model Prediction...")
    rf_predictions, rf_y_test = pm.predict_with_random_forest(df)

    # XGBoost Model
    logging.info("XGBoost Model Prediction...")
    xgb_predictions, xgb_y_test = pm.predict_with_xgboost(df)

    # SVM Model
    logging.info("SVM Model Prediction...")
    svm_predictions, svm_y_test = pm.predict_with_svm(df)

    # CNN Model
    logging.info("CNN Model Prediction...")
    cnn_predictions, cnn_y_test = pm.predict_with_cnn(df)

    # Transformer Model
    logging.info("Transformer Model Prediction...")
    transformer_predictions, transformer_y_test = pm.predict_with_transformer(df)

    # Step 4: Backtesting
    logging.info("Step 4: Backtesting Models...")

    # For ARIMA and Prophet (forecasting next 30 days)
    actual_prices_arima_prophet = df['Close'].iloc[-30:].values
    arima_forecast_values = arima_forecast.values
    prophet_predicted_prices = prophet_forecast.set_index('ds')['yhat'].iloc[-30:].values

    # Ensure lengths match
    min_length = min(len(actual_prices_arima_prophet), len(arima_forecast_values), len(prophet_predicted_prices))
    actual_prices_arima_prophet = actual_prices_arima_prophet[-min_length:]
    arima_forecast_values = arima_forecast_values[-min_length:]
    prophet_predicted_prices = prophet_predicted_prices[-min_length:]

    # Calculate error metrics
    arima_errors = bt.calculate_error_metrics(actual_prices_arima_prophet, arima_forecast_values)
    prophet_errors = bt.calculate_error_metrics(actual_prices_arima_prophet, prophet_predicted_prices)
    lstm_errors = bt.calculate_error_metrics(y_test, lstm_predictions)

    # SARIMA Model Backtesting
    actual_prices_sarima = df['Close'].iloc[-30:].values
    sarima_forecast_values = sarima_forecast.values

    min_length_sarima = min(len(actual_prices_sarima), len(sarima_forecast_values))
    actual_prices_sarima = actual_prices_sarima[-min_length_sarima:]
    sarima_forecast_values = sarima_forecast_values[-min_length_sarima:]

    sarima_errors = bt.calculate_error_metrics(actual_prices_sarima, sarima_forecast_values)

    # Random Forest Errors
    rf_errors = bt.calculate_error_metrics(rf_y_test, rf_predictions)

    # XGBoost Errors
    xgb_errors = bt.calculate_error_metrics(xgb_y_test, xgb_predictions)

    # SVM Errors
    svm_errors = bt.calculate_error_metrics(svm_y_test, svm_predictions)

    # CNN Errors
    cnn_errors = bt.calculate_error_metrics(cnn_y_test, cnn_predictions)

    # Transformer Errors
    transformer_errors = bt.calculate_error_metrics(transformer_y_test, transformer_predictions)

    # Display error metrics
    logging.info("\nARIMA Model Error Metrics:")
    bt.display_error_metrics(*arima_errors)

    logging.info("\nProphet Model Error Metrics:")
    bt.display_error_metrics(*prophet_errors)

    logging.info("\nLSTM Model Error Metrics:")
    bt.display_error_metrics(*lstm_errors)

    logging.info("\nSARIMA Model Error Metrics:")
    bt.display_error_metrics(*sarima_errors)

    logging.info("\nRandom Forest Model Error Metrics:")
    bt.display_error_metrics(*rf_errors)

    logging.info("\nXGBoost Model Error Metrics:")
    bt.display_error_metrics(*xgb_errors)

    logging.info("\nSVM Model Error Metrics:")
    bt.display_error_metrics(*svm_errors)

    logging.info("\nCNN Model Error Metrics:")
    bt.display_error_metrics(*cnn_errors)

    logging.info("\nTransformer Model Error Metrics:")
    bt.display_error_metrics(*transformer_errors)

    # Step 5: Risk Assessment
    logging.info("Step 5: Performing Risk Assessment...")
    df = ra.calculate_daily_returns(df)
    ra.plot_return_distribution(df, ticker_symbol)
    volatility, ann_return, sharpe_ratio = ra.calculate_risk_metrics(df)
    utils.print_summary(f'Volatility: {volatility:.4f}\nAnnual Return: {ann_return:.4f}\nSharpe Ratio: {sharpe_ratio:.4f}')

    # Optional: Plotting LSTM Predictions
    logging.info("Plotting LSTM Predictions...")
    plot_lstm_results(df, y_test, lstm_predictions, ticker_symbol)

    # Step 6: Generate Insights with GPT
    logging.info("Step 6: Generating Insights with GPT-4o...")
    system_prompt = (    """
You are a financial analyst with a strong background in machine learning-based stock analysis and trading strategies. You are tasked with providing detailed insights and actionable recommendations based on the provided stock analysis data, which includes performance metrics from several predictive models, risk assessments, and market conditions.

Your primary goal is to help the user make a well-informed decision on whether to buy, sell, or hold the stock. Consider the following steps when generating your response:

1. **Review Model Performance Metrics**:
    
    - Analyze the predictive performance of models (ARIMA, LSTM, Random Forest, etc.) using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
    - Identify the models that have the most accurate predictions and explain the reliability of their forecasts over different time frames (short-term vs long-term).
2. **Evaluate Risk and Volatility**:
    
    - Review risk metrics such as volatility, annual returns, and the Sharpe ratio to assess the risk/reward profile of the stock.
    - Identify potential risks in the stock’s behavior, such as high volatility, large fluctuations in predictions, or inconsistency in historical patterns.
    - Comment on how these risk factors may affect the stock's future performance.
3. **Identify Market Trends**:
    
    - Examine the stock’s recent performance, momentum, and trend direction based on the predictions.
    - Highlight whether the stock is showing signs of growth, stagnation, or decline and whether these trends are consistent across different models.
    - Predict future movements of the stock based on model forecasts and historical trends.
4. **Make a Buy/Sell/Hold Recommendation**:
    
    - Based on the combined analysis of model performance and risk assessment, provide a clear and actionable recommendation on whether to buy, sell, or hold the stock.
    - Explain the reasoning behind your recommendation. For example, “Buy because the stock is undervalued with low volatility” or “Sell because of high volatility and uncertain trends.”
    - If the recommendation is to hold, explain why it might be best to wait for further confirmation of trends or market conditions.
5. **Consider Broader Market Context**:
    
    - Factor in market conditions and how they might impact the stock’s performance. If applicable, provide a context-driven explanation of how external factors like market volatility, earnings reports, or macroeconomic trends might influence the stock.

Your final output should provide clear, structured, and actionable advice, backed by data-driven insights, that can assist the user in making a well-informed investment decision.
                     """

    )

    user_prompt = f"""
    Stock Analysis Results for {ticker_symbol} from {start_date} to {end_date}:

    ARIMA Model Error Metrics:
    - MAE: {arima_errors[0]:.2f}
    - MSE: {arima_errors[1]:.2f}
    - RMSE: {arima_errors[2]:.2f}

    Prophet Model Error Metrics:
    - MAE: {prophet_errors[0]:.2f}
    - MSE: {prophet_errors[1]:.2f}
    - RMSE: {prophet_errors[2]:.2f}

    LSTM Model Error Metrics:
    - MAE: {lstm_errors[0]:.2f}
    - MSE: {lstm_errors[1]:.2f}
    - RMSE: {lstm_errors[2]:.2f}

    SARIMA Model Error Metrics:
    - MAE: {sarima_errors[0]:.2f}
    - MSE: {sarima_errors[1]:.2f}
    - RMSE: {sarima_errors[2]:.2f}

    Random Forest Model Error Metrics:
    - MAE: {rf_errors[0]:.2f}
    - MSE: {rf_errors[1]:.2f}
    - RMSE: {rf_errors[2]:.2f}

    XGBoost Model Error Metrics:
    - MAE: {xgb_errors[0]:.2f}
    - MSE: {xgb_errors[1]:.2f}
    - RMSE: {xgb_errors[2]:.2f}

    SVM Model Error Metrics:
    - MAE: {svm_errors[0]:.2f}
    - MSE: {svm_errors[1]:.2f}
    - RMSE: {svm_errors[2]:.2f}

    CNN Model Error Metrics:
    - MAE: {cnn_errors[0]:.2f}
    - MSE: {cnn_errors[1]:.2f}
    - RMSE: {cnn_errors[2]:.2f}

    Transformer Model Error Metrics:
    - MAE: {transformer_errors[0]:.2f}
    - MSE: {transformer_errors[1]:.2f}
    - RMSE: {transformer_errors[2]:.2f}

    Risk Metrics:
    - Volatility: {volatility:.4f}
    - Annual Return: {ann_return:.4f}
    - Sharpe Ratio: {sharpe_ratio:.4f}

    Based on these results, please provide insights and suggestions.
    """

    # Generate insights
    insights = generate_insights(system_prompt, user_prompt)
    
    # Display insights
    logging.info("\nGenerated Insights:")
    logging.info(insights)
    
    # Optionally, save insights to a file
    with open(f'{ticker_symbol}_Insights.txt', 'w') as f:
        f.write(insights)
    
def plot_lstm_results(df, y_test, lstm_predictions, ticker_symbol):
    # Prepare data for plotting
    train_data_len = len(df) - len(y_test)
    train = df[['Close']].iloc[:train_data_len]
    valid = df[['Close']].iloc[train_data_len:].copy()
    valid['Predictions'] = lstm_predictions
    
    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.title(f'{ticker_symbol} Closing Price Prediction using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'], label='Training Data')
    plt.plot(valid['Close'], label='Actual Price')
    plt.plot(valid['Predictions'], label='Predicted Price')
    plt.legend(loc='upper left')
    plt.savefig(f'{ticker_symbol}_LSTM_Predictions.png')
    plt.close()
    logging.info(f'LSTM prediction plot saved as {ticker_symbol}_LSTM_Predictions.png')

if __name__ == '__main__':
    main()
