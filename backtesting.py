from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import logging

def calculate_error_metrics(y_true, y_pred):
    logging.info("Calculating error metrics...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def display_error_metrics(mae, mse, rmse):
    logging.info(f'Mean Absolute Error (MAE): {mae}')
    logging.info(f'Mean Squared Error (MSE): {mse}')
    logging.info(f'Root Mean Squared Error (RMSE): {rmse}')
