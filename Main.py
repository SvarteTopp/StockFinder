import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Fetch historical stock data
def fetch_data(ticker, period="5y"):
    data = yf.download(ticker, period=period)
    print(data.columns)  #Debugging step
    if 'Adj Close' not in data.columns: #Use 'Close' if 'Adj Close' is missing
        data['Adj Close'] = data['Close']
    data['Return'] = data['Adj Close'].pct_change()
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
    data['Volatility'] = data['Return'].rolling(window=20).std()
    data = data.dropna()
    return data

# Step 2: Feature engineering
def prepare_features(data):
    data['Signal'] = np.where(data['SMA_20'] > data['SMA_50'], 1, 0)  # 1: Buy signal
    features = ['SMA_20', 'SMA_50', 'Volatility', 'Return']
    return data[features], data['Signal']

# Step 3: Train a model
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model

# Step 4: Predict buy signals
def predict_signals(model, features):
    predictions = model.predict(features)
    return predictions

# Main script
if __name__ == "__main__":
    ticker = "AAPL"  # Example stock
    data = fetch_data(ticker)
    features, target = prepare_features(data)
    
    print("Training model...")
    model = train_model(features, target)

    print("Generating predictions...")
    data['Prediction'] = predict_signals(model, features)
    
    print("Buy signals:")
    print(data[data['Prediction'] == 1].tail())
