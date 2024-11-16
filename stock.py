import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the stock ticker and date range
ticker = "MSFT"  # Microsoft
start_date = "2023-01-01"
end_date = "2024-11-5"

# Download the data
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate various formulas for prepro
data["MA-5"] = data["Close"].rolling(window=5).mean() 
data["MA-20"] = data["Close"].rolling(window=20).mean() 
data["MA-100"] = data["Close"].rolling(window=100).mean() 
data["PC"] = data["Close"].pct_change() * 100
data["volatility"] = data["Close"].rolling(window = 20).std()  * (20 ** 0.5)  
data = data.dropna()

########################################################
# Data adjustments
########################################################

# Reset the index to make Date a column 
data = data.reset_index()

#all for one ahh features
features = data[['Close', 'MA-5', 'MA-20', 'MA-100', 'Volume', "PC", "volatility"]]

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler.fit_transform(features[['Close', 'MA-5', 'MA-20', 'MA-100', 'Volume', "PC", "volatility"]])

scaled_features_df = pd.DataFrame(scaled_features, columns=['Close', 'MA-5', 'MA-20', 'MA-100', 'Volume', 'PC', "volatility"])


length = 30
data_values = scaled_features_df.values
X, y = [], []

for i in range(len(data_values) - length):
     X.append(data_values[i:i+length])
     y.append(data_values[i+length, 0])

X = np.array(X)
y = np.array(y)

print(f"Input shape (X): {X.shape}")
print(f"Output shape (y): {y.shape}")


########################################################
# Model Training
########################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
 
# Define the model
model = Sequential()

# Add LSTM layer
model.add(LSTM(units=50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Add another LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add Dense output layer
model.add(Dense(units=1))  

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())

history = model.fit(X_train, y_train, epochs = 50, batch_size = 16, validation_data = (X_test, y_test), verbose = 1)

#print loss plot
'''
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
'''
########################################################
# prediction 
########################################################

last_sequence = scaled_features_df[-length:].values

# Initialize predictions list
predicted_values = []

for _ in range(10):
    # Reshape to match model input
    input_sequence = np.expand_dims(last_sequence, axis=0)  
    
    # Predict the next day
    next_day_prediction = model.predict(input_sequence)
    
    # Append the prediction to the list
    predicted_values.append(next_day_prediction[0][0])
    
    # Update the sequence for the next prediction
    next_day_data = np.append(last_sequence[1:], [[next_day_prediction[0][0]] + list(last_sequence[-1][1:])], axis=0)
    last_sequence = next_day_data

# Inverse transform predictions to get the actual values
predicted_values_unscaled = scaler.inverse_transform(
    [[pred] + [0] * (scaled_features_df.shape[1] - 1) for pred in predicted_values]
)[:, 0]


# Print the predictions for November 6th to 15th
print("Predicted Stock Prices (November 6th to 15th):")
for pred in (predicted_values_unscaled):
    print(f"Predicted Close Price: {pred:.2f}")


########################################################
# Next weeks actual data
########################################################

next_week_start_date = "2024-11-6"
next_week_end_date = "2024-11-15"
next_week_data = yf.download(ticker, start = next_week_start_date, end = next_week_end_date)

# Extract the adjusted close price
actual_next_week_prices = next_week_data['Close'].values
print(actual_next_week_prices)



