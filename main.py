import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score

API_KEY = 'zIikXy77dcwqLwEl5enlwQBOQharcSzq'

def fetch_stock_data(stock_symbol, start_date, end_date):
    """Fetch stock data from Polygon.io."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{stock_symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": "true", 
        "sort": "asc", 
        "limit": 50000, 
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()["results"]
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["t"], unit='ms')  # Convert timestamp to date
        df.rename(columns={"o": "Open", "h": "High", "l": "Low", 
                           "c": "Close", "v": "Volume"}, inplace=True)
        return df
    else:
        print(f"Failed to fetch data for {stock_symbol}: {response.json()}")
        return None

stock_dict = {
    "TSLA": "Tesla",
    "GOOGL": "Google"
}

for stock_symbol, stock_name in stock_dict.items():
    # Fetch data from Polygon.io
    df = fetch_stock_data(stock_symbol, start_date="2022-01-01", end_date="2024-10-14")
    
    if df is None:
        continue  # Skip if data fetch failed

    # Display date range and days in dataset
    print(f'Dataframe contains stock prices between {df.Date.min()} and {df.Date.max()}')
    print(f'Total days = {(df.Date.max() - df.Date.min()).days} days')

    # Plot a boxplot of the stock prices
    df[['Open', 'High', 'Low', 'Close']].plot(kind='box', figsize=(10, 6))
    plt.title(f'{stock_name} Stock Prices Boxplot')
    plt.show()

    # Plot closing price over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Closing Price')
    plt.title(f'{stock_name} Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Prepare data for the linear regression model
    X = np.array(df.index).reshape(-1, 1)
    Y = df['Close']

    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

    # Feature scaling
    scaler = StandardScaler().fit(X_train)

    # Create and train the linear regression model
    lm = LinearRegression()
    lm.fit(X_train, Y_train)

    # Plot actual vs predicted values for the training set
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train, Y_train, color='blue', label='Actual', alpha=0.6)
    plt.plot(X_train, lm.predict(X_train), color='red', label='Predicted')
    plt.title('Actual vs Predicted - Training Data')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Calculate and print evaluation metrics
    train_r2 = r2_score(Y_train, lm.predict(X_train))
    test_r2 = r2_score(Y_test, lm.predict(X_test))
    train_mse = mse(Y_train, lm.predict(X_train))
    test_mse = mse(Y_test, lm.predict(X_test))

    scores = f'''
    {'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
    {'r2_score'.ljust(10)}{train_r2:<20}{test_r2}
    {'MSE'.ljust(10)}{train_mse:<20}{test_mse}
    '''
    print(scores)
