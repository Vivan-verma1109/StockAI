import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the stock ticker and date range
ticker = "MSFT"  # Microsoft
start_date = "2023-01-01"
end_date = "2024-11-5"

# Download the data
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate various formulas for prepro
data["MA-5"] = data["Adj Close"].rolling(window=5).mean() 
data["MA-20"] = data["Adj Close"].rolling(window=20).mean() 
data["MA-100"] = data["Adj Close"].rolling(window=100).mean() 
data["PC"] = data["Adj Close"].pct_change() * 100
data["volatility"] = data["Adj Close"].rolling(window = 20).std()  * (20 ** 0.5)  
data = data.dropna()

########################################################
# Data adjustments
########################################################

# Reset the index to make Date a column 
data = data.reset_index()

#all for one ahh features
features = data[['Adj Close', 'MA-5', 'MA-20', 'MA-100', 'Volume', "PC", "volatility"]]

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler.fit_transform(features[['Adj Close', 'MA-5', 'MA-20', 'MA-100', 'Volume', "PC", "volatility"]])

scaled_features_df = pd.DataFrame(scaled_features, columns=['Adj Close', 'MA-5', 'MA-20', 'MA-100', 'Volume', 'PC', "volatility"])


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
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Add another LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add Dense output layer
model.add(Dense(units=1))  

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())

history = model.fit(X_train, y_train, epochs = 50, batch_size = 32, validation_data = (X_test, y_test), verbose = 1)

#print loss plot
'''
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
'''
########################################################
# Next weeks actual data
########################################################

# Update the date range to get the next week's actual data



########################################################
# Im gonna actually try to predict this now
########################################################
