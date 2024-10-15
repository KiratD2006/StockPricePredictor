import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For regression and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score

# Load dataset
tesla = pd.read_csv('tesla.csv')
tesla['Date'] = pd.to_datetime(tesla['Date'], dayfirst=True)

# Display date range and days in dataset
print(f'Dataframe contains stock prices between {tesla.Date.min()} and {tesla.Date.max()}')
print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days} days')

# Plot a boxplot of the stock prices
tesla[['Open', 'High', 'Low', 'Close', 'Adj Close']].plot(kind='box', figsize=(10, 6))
plt.title('Tesla Stock Prices Boxplot')
plt.show()

# Plot closing price over time
plt.figure(figsize=(12, 6))
plt.plot(tesla['Date'], tesla['Close'], label='Closing Price')
plt.title('Tesla Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Prepare data for the linear regression model
X = np.array(tesla.index).reshape(-1, 1)
Y = tesla['Close']

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
