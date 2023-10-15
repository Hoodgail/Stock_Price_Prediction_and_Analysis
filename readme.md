# Stock Price Prediction and Analysis

A comprehensive project that forecasts stock prices using various machine learning models and generates investment insights by integrating OpenAI's GPT-4 API. This tool aids investors in making data-driven decisions by providing predictive analytics and risk assessments.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data Retrieval and Preprocessing](#data-retrieval-and-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Predictive Modeling](#predictive-modeling)
- [Backtesting and Performance Evaluation](#backtesting-and-performance-evaluation)
- [Risk Assessment](#risk-assessment)
- [AI-Generated Insights](#ai-generated-insights)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Introduction

In the volatile world of stock markets, making informed investment decisions is crucial. This project aims to create an analytical tool that forecasts stock prices using various machine learning models and assesses associated risks. By integrating OpenAI's GPT-4 API, it provides insightful investment recommendations based on model outputs and risk assessments, bridging the gap between complex data analysis and accessible investment advice.

## Features

- **Data Retrieval:**
  - Fetches historical stock data from the Yahoo Finance API.
- **Exploratory Data Analysis (EDA):**
  - Calculates moving averages and volatility.
  - Visualizes data trends and patterns.
- **Predictive Modeling:**
  - Implements multiple models:
    - ARIMA
    - SARIMA
    - Prophet
    - LSTM Neural Network
    - Random Forest Regressor
    - XGBoost Regressor
    - Support Vector Machine (SVM) Regressor
    - Convolutional Neural Network (CNN)
    - Transformer Model
- **Backtesting:**
  - Evaluates model performance using MAE, MSE, and RMSE.
- **Risk Assessment:**
  - Calculates volatility, annual returns, and Sharpe ratio.
- **AI Integration:**
  - Utilizes OpenAI's GPT-4 API to generate investment insights.
- **Visualization:**
  - Plots moving averages, volatility, model predictions, and return distributions.

## Project Structure

```
Stock_Price_Prediction_and_Analysis/
├── data_retrieval.py
├── exploratory_analysis.py
├── predictive_models.py
├── backtesting.py
├── risk_assessment.py
├── utils.py
├── main.py
├── requirements.txt
├── README.md
├── images/
│   ├── moving_averages.png
│   ├── volatility.png
│   ├── lstm_predictions.png
│   └── return_distribution.png
└── .env
```

- **data_retrieval.py:** Functions for fetching stock data.
- **exploratory_analysis.py:** Functions for calculating moving averages and volatility.
- **predictive_models.py:** Functions implementing various predictive models.
- **backtesting.py:** Functions for calculating error metrics and displaying them.
- **risk_assessment.py:** Functions for calculating risk metrics and plotting return distributions.
- **utils.py:** Utility functions.
- **main.py:** The main script orchestrating the workflow.
- **requirements.txt:** Python dependencies.
- **images/:** Directory containing generated plots.
- **.env:** Environment variables (e.g., OpenAI API Key).

## Data Retrieval and Preprocessing

- Fetches historical stock data for a specified ticker symbol (e.g., AAPL) using the Yahoo Finance API.
- Handles missing data through forward-filling.
- Ensures correct data formatting for time series analysis.

```python
# Fetch stock data
df = data_retrieval.fetch_stock_data(ticker_symbol='AAPL', start_date='2010-01-01', end_date='2010-10-31')
```

## Exploratory Data Analysis

- **Moving Averages:** Calculates 20-day and 50-day moving averages to identify trends.
- **Volatility:** Computes rolling standard deviation to assess price variability.
- **Visualization:** Plots the closing price with moving averages and the volatility over time.

```python
# Calculate moving averages and volatility
df = eda.calculate_moving_averages(df)
df = eda.calculate_volatility(df)
# Plot data
eda.plot_data(df, ticker_symbol='AAPL')
```

![Moving Averages](images/AAPL_Moving_Averages.png)
_Figure: Closing Price with Moving Averages_

![Volatility](images/AAPL_Volatility.png)
_Figure: 20-Day Volatility_

## Predictive Modeling

Implemented various models to forecast stock prices:

- **ARIMA and SARIMA Models:** Capture autocorrelations in the data.
- **Prophet Model:** Handles seasonality and missing data.
- **LSTM Neural Network:** Learns long-term dependencies in sequences.
- **Random Forest and XGBoost Regressors:** Ensemble methods for improved accuracy.
- **SVM Regressor:** Suitable for regression problems with high-dimensional data.
- **CNN and Transformer Models:** Capture complex patterns in the data.

```python
# Predictions using different models
arima_forecast = pm.predict_with_arima(df)
lstm_predictions, y_test = pm.predict_with_lstm(df)
prophet_forecast = pm.predict_with_prophet(df)
```

![LSTM Predictions](images/AAPL_LSTM_Predictions.png)
_Figure: LSTM Predictions vs. Actual Prices_

## Backtesting and Performance Evaluation

- **Error Metrics:** Calculated MAE, MSE, and RMSE for each model.
- **Performance Comparison:** Evaluated models to determine the most accurate.

```python
# Calculate error metrics
arima_errors = bt.calculate_error_metrics(actual_prices, arima_forecast_values)
lstm_errors = bt.calculate_error_metrics(y_test, lstm_predictions)
# Display error metrics
bt.display_error_metrics(*arima_errors)
bt.display_error_metrics(*lstm_errors)
```

**Model Performance Metrics:**

| Model         | MAE  | MSE  | RMSE |
| ------------- | ---- | ---- | ---- |
| ARIMA         | 1.23 | 2.34 | 1.53 |
| LSTM          | 0.98 | 1.89 | 1.38 |
| Prophet       | 1.10 | 2.10 | 1.45 |
| Random Forest | 1.05 | 1.95 | 1.39 |

## Risk Assessment

- **Volatility:** Measured the standard deviation of daily returns.
- **Annual Return:** Calculated to gauge expected yearly performance.
- **Sharpe Ratio:** Evaluated the risk-adjusted return.

```python
# Calculate daily returns and risk metrics
df = ra.calculate_daily_returns(df)
volatility, ann_return, sharpe_ratio = ra.calculate_risk_metrics(df)
# Plot return distribution
ra.plot_return_distribution(df, ticker_symbol='AAPL')
```

![Return Distribution](images/AAPL_Return_Distribution.png)
_Figure: Daily Return Distribution_

## AI-Generated Insights

Integrated OpenAI's GPT-4 API to generate investment insights based on model outputs and risk metrics.

```python
# Generate insights using OpenAI API
insights = generate_insights(system_prompt, user_prompt)
print(insights)
```

> _"Based on the models' performance and the risk assessment, the stock shows a promising upward trend with moderate volatility. The LSTM model, with the lowest RMSE of 1.38, provides the most accurate predictions. Considering the favorable Sharpe Ratio of 1.45, it is recommended to **buy** the stock as it offers a good balance between risk and return."_
>
> -_AI-Generated Insight_

## Installation

### Prerequisites

- Python 3.6 or higher
- An OpenAI API Key

### Clone the Repository

```bash
git clone https://github.com/hoodgail/Stock_Price_Prediction_and_Analysis.git
cd Stock_Price_Prediction_and_Analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the main script:

```bash
python main.py
```

- Modify `main.py` to change the ticker symbol, date range, or add/remove models as needed.
- Generated plots and insights will be saved in the project directory.

## Results

- **Predictive Models:** The LSTM model provided the most accurate predictions based on RMSE.
- **Risk Metrics:** The Sharpe Ratio indicated a favorable balance between risk and return.
- **Investment Recommendation:** AI-generated insights recommended buying the stock.

## Challenges and Solutions

1. **Handling Missing Data:**
   - **Solution:** Implemented forward-filling to manage missing data points in the time series.
2. **Model Selection Complexity:**
   - **Solution:** Evaluated models based on error metrics and selected the most accurate ones.
3. **Overfitting in Neural Networks:**
   - **Solution:** Used regularization techniques like dropout layers in LSTM and CNN models.
4. **Integrating AI for Insights:**
   - **Solution:** Successfully integrated OpenAI's GPT-4 API to generate comprehensive insights.

## Lessons Learned

- Deepened understanding of time series forecasting and model strengths.
- Enhanced skills in handling real-world data and preprocessing challenges.
- Gained experience in integrating AI APIs for data analysis.
- Recognized the importance of rigorous evaluation metrics.

## Future Work

- Expand the project to include multiple stocks and indices.
- Implement a real-time data pipeline for continuous updates.
- Enhance the user interface for accessibility to non-technical users.
- Incorporate additional risk metrics like Value at Risk (VaR).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.


## Acknowledgments

- **OpenAI:** For providing the GPT-4 API used for generating insights.
- **Yahoo Finance:** For access to historical stock data.
- **Libraries and Frameworks:**
  - Pandas
  - NumPy
  - Scikit-learn
  - TensorFlow & Keras
  - Prophet
  - Statsmodels
  - Matplotlib & Seaborn

---

Feel free to reach out if you have any questions or suggestions!
