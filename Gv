import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Read data from Excel file
df = pd.read_excel('your_excel_file.xlsx', index_col=0)  # Assuming ASINs are in the first column

# Calculate seasonal indices
seasonal_indices = df.mean(axis=1) / df.mean().mean()  # Calculate average page hits for each week and derive seasonal indices

# Initialize dictionaries to store forecasts and error metrics
forecasts = {}  # Dictionary to store forecasted page hits for each ASIN
mae_scores = {}  # Dictionary to store Mean Absolute Error (MAE) scores for each ASIN
mse_scores = {}  # Dictionary to store Mean Squared Error (MSE) scores for each ASIN

# Loop over each ASIN
for asin in df.index:
    asin_data = df.loc[asin].dropna()  # Extract page hit data for the current ASIN and remove missing values
    deseasonalized_data = asin_data / seasonal_indices  # Deseasonalize data by dividing by seasonal indices

    train_data = deseasonalized_data[:-1]  # Use all data except the last week for training
    test_data = deseasonalized_data[-1:]   # Use the last week for testing

    # Fit ARIMA model
    model = ARIMA(train_data, order=(5,1,0))  # Fit ARIMA model with specified order (p=5, d=1, q=0)
    fit_model = model.fit()  # Fit the ARIMA model to the training data

    # Forecast
    forecast = fit_model.forecast(steps=1)[0]  # Make a one-step forecast
    forecasts[asin] = round(forecast * seasonal_indices[-1])  # Reseasonalize forecast by multiplying with seasonal index of last week

    # Calculate error metrics
    mae = mean_absolute_error(test_data, forecast)  # Calculate Mean Absolute Error (MAE)
    mse = mean_squared_error(test_data, forecast)  # Calculate Mean Squared Error (MSE)
    mae_scores[asin] = mae  # Store MAE score for current ASIN
    mse_scores[asin] = mse  # Store MSE score for current ASIN

# Print forecasts and error metrics
for asin, forecast in forecasts.items():
    print(f"Forecasted page hits for {asin} in week 7:", forecast)  # Print forecasted page hits for each ASIN

print("\nMean Absolute Error (MAE) Scores:")
for asin, mae in mae_scores.items():
    print(f"MAE for {asin}: {mae}")  # Print MAE scores for each ASIN

print("\nMean Squared Error (MSE) Scores:")
for asin, mse in mse_scores.items():
    print(f"MSE for {asin}: {mse}")  # Print MSE scores for each ASIN
