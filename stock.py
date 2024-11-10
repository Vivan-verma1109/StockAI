import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the stock ticker and date range
ticker = "AAPL"  # Apple Inc.
start_date = "2020-11-09"
end_date = "2024-11-09"

# Download the data
data = yf.download(ticker, start=start_date, end=end_date)

# Isolate the Adj Close column
close_data = data[['Adj Close']].values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)  # Rescale to 0 - 1

# Define window size for sequences
window_size = 30

# Prepare training data
X_train, y_train = [], []
for i in range(window_size, len(scaled_data)):
    X_train.append(scaled_data[i - window_size:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initialize the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs = 20, batch_size = 8)

# Preparing data for future predictions
last_window = scaled_data[-window_size:]  # Last 30 days to predict future prices

# Predicting the next 7 days
predictions = []
for _ in range(7):
    X_test = np.array([last_window])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price = model.predict(X_test)
    predictions.append(predicted_price[0, 0])  # Store prediction
    
    # Update the window with the latest prediction
    last_window = np.append(last_window[1:], predicted_price[0, 0]).reshape(-1, 1)

# Inverse transform the predictions back to original scale
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

print("Predicted prices for the next 7 days:")
print(predicted_prices)
