import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the stock ticker and date range
ticker = "MSFT"  # Microsoft
start_date = "2023-01-01"
end_date = "2024-11-10"

# Download the data
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate moving averages
data["MA-5"] = data["Adj Close"].rolling(window=5).mean() 
data["MA-20"] = data["Adj Close"].rolling(window=20).mean() 
data["MA-100"] = data["Adj Close"].rolling(window=100).mean() 

# Reset the index to make Date a column 
data = data.reset_index()

features = data[['Adj Close', 'MA-5', 'MA-20', 'MA-100', 'Volume']]

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler.fit_transform(features[['Adj Close', 'MA-5', 'MA-20', 'MA-100', 'Volume']])

scaled_features_df = pd.DataFrame(scaled_features, columns=['Adj Close', 'MA-5', 'MA-20', 'MA-100', 'Volume'])
