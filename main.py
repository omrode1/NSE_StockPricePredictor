import pandas as pd
import matplotlib.pyplot as plt
import joblib
import argparse
import os

def load_data(file_path):
    """Load the input CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the input data."""
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Create moving averages
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    
    # Fill missing values
    data = data.fillna(method='ffill')
    
    return data

def make_predictions(data, model):
    """Make predictions using the trained model."""
    X_new = data[['MA20', 'MA50', 'MA100']]
    predictions = model.predict(X_new)
    data['Predicted_Close'] = predictions
    return data

def plot_results(data):
    """Plot the actual vs predicted prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Actual Prices')
    plt.plot(data['Date'], data['Predicted_Close'], label='Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def main(input_file, output_file):
    # Load the input data
    data = load_data(input_file)
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Load the trained model
    model = joblib.load('stock_price_model.pkl')
    
    # Make predictions
    data = make_predictions(data, model)
    
    # Save the results
    data.to_csv(output_file, index=False)
    
    # Plot the results
    plot_results(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV file')
    args = parser.parse_args()
    
    main(args.input, args.output)
