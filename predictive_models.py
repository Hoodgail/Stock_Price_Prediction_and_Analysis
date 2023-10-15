from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras import layers, models
import numpy as np
import pandas as pd
import logging
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.dates as mpl_dates

LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D = layers.LayerNormalization, layers.MultiHeadAttention, layers.GlobalAveragePooling1D


Sequential = models.Sequential
Dense = layers.Dense
LSTM = layers.LSTM
Dropout = layers.Dropout

def predict_with_arima(df):
    logging.info("Building and fitting ARIMA model...")
    # Ensure the index has a frequency for ARIMA
    df_arima = df.copy()
    df_arima = df_arima.asfreq('B')  # Set frequency to business days
    df_arima['Close'] = df_arima['Close'].ffill()  # Forward fill to handle missing data
    model = ARIMA(df_arima['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    logging.info("ARIMA model forecasting completed.")
    return forecast

def predict_with_sarima(df):
    logging.info("Building and fitting SARIMA model...")
    # Prepare the data
    df_sarima = df.copy()
    df_sarima = df_sarima.asfreq('B')  # Set frequency to business days
    df_sarima['Close'] = df_sarima['Close'].ffill()  # Forward fill missing data

    # Define the SARIMA model parameters
    order = (1, 1, 1)  # ARIMA parameters
    seasonal_order = (1, 1, 1, 12)  # Seasonal parameters

    # Fit the SARIMA model
    model = SARIMAX(df_sarima['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
    logging.info("SARIMA model forecasting completed.")
    return forecast

def predict_with_prophet(df):
    logging.info("Preparing data for Prophet model...")
    # Prepare the data
    prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    # Remove timezone information
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    model = Prophet(daily_seasonality=True)
    logging.info("Fitting Prophet model...")
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    logging.info("Prophet model forecasting completed.")
    return forecast

def predict_with_garch(df):
    logging.info("Building and fitting GARCH model...")
    # Calculate returns
    returns = df['Close'].pct_change().dropna() * 100  # Percentage returns

    # Fit the GARCH(1,1) model
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    # Forecast volatility
    forecast = model_fit.forecast(horizon=30)
    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
    logging.info("GARCH model forecasting completed.")
    return volatility_forecast

def predict_with_lstm(df):
    logging.info("Preparing data for LSTM model...")
    # Prepare the data
    data = df.filter(['Close']).copy()
    data.dropna(inplace=True)
    dataset = data.values

    # Define the training data length
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]

    # Split the training data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data to 3D for LSTM input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    logging.info("Building the LSTM model...")
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    logging.info("Training the LSTM model...")
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape x_test for LSTM input
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Generate predictions
    logging.info("Generating predictions with the LSTM model...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    logging.info("LSTM model prediction completed.")
    return predictions, y_test

def predict_with_random_forest(df):
    logging.info("Building and fitting Random Forest Regressor...")
    # Prepare the data
    df_rf = df.copy()
    df_rf['Date'] = df_rf.index.map(mpl_dates.date2num)  # Convert dates to numerical format
    df_rf = df_rf[['Date', 'Close']].dropna()

    X = df_rf[['Date']]
    y = df_rf['Close']

    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Instantiate and train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    predictions = rf_model.predict(X_test)

    logging.info("Random Forest model prediction completed.")
    return predictions, y_test

def predict_with_xgboost(df):
    logging.info("Building and fitting XGBoost Regressor...")
    # Prepare the data
    df_xgb = df.copy()
    df_xgb['Date'] = df_xgb.index.map(mpl_dates.date2num)
    df_xgb = df_xgb[['Date', 'Close']].dropna()

    X = df_xgb[['Date']]
    y = df_xgb['Close']

    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Instantiate and train the model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Make predictions
    predictions = xgb_model.predict(X_test)

    logging.info("XGBoost model prediction completed.")
    return predictions, y_test

def predict_with_svm(df):
    logging.info("Building and fitting Support Vector Regression model...")
    # Prepare the data
    df_svm = df.copy()
    df_svm['Date'] = df_svm.index.map(mpl_dates.date2num)
    df_svm = df_svm[['Date', 'Close']].dropna()

    X = df_svm[['Date']]
    y = df_svm['Close']

    # Split data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Instantiate and train the model
    svm_model = SVR(kernel='rbf', C=1000.0, gamma=0.1)
    svm_model.fit(X_train, y_train)

    # Make predictions
    predictions = svm_model.predict(X_test)

    logging.info("SVM model prediction completed.")
    return predictions, y_test

def predict_with_cnn(df):
    logging.info("Preparing data for CNN model...")
    # Prepare the data similar to LSTM
    data = df.filter(['Close']).copy()
    data.dropna(inplace=True)
    dataset = data.values

    training_data_len = int(np.ceil(len(dataset) * 0.8))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data for CNN input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the CNN model
    logging.info("Building the CNN model...")
    model = Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    logging.info("Training the CNN model...")
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    # Prepare the testing data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Generate predictions
    logging.info("Generating predictions with the CNN model...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    logging.info("CNN model prediction completed.")
    return predictions, y_test

def predict_with_transformer(df):
    logging.info("Preparing data for Transformer model...")
    # Prepare the data similar to LSTM
    data = df.filter(['Close']).copy()
    data.dropna(inplace=True)
    dataset = data.values

    training_data_len = int(np.ceil(len(dataset) * 0.8))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data for Transformer input
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the Transformer model
    logging.info("Building the Transformer model...")
    input_layer = layers.Input(shape=(x_train.shape[1], 1))
    x = LayerNormalization()(input_layer)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(50, activation='relu')(x)
    output_layer = Dense(1)(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    logging.info("Training the Transformer model...")
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)

    # Prepare the testing data
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Generate predictions
    logging.info("Generating predictions with the Transformer model...")
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    logging.info("Transformer model prediction completed.")
    return predictions, y_test
