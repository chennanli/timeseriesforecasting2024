import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
from src.config import PROCESSED_DATA_DIR
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# Print current working directory and script location
print("Current working directory:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))

# Print Python path
print("Python path:")
for path in sys.path:
    print(path)




# Load the processed data for NXT
data_path = os.path.join(PROCESSED_DATA_DIR, 'prepared_data_NXT.csv')
df = pd.read_csv(data_path, parse_dates=['Date'])

# Sort the DataFrame by the 'Date' column
df = df.sort_values('Date')

# Set 'Date' as the index after sorting
df.set_index('Date', inplace=True)

print("First few rows of the data:")
print(df.head())

print("\nData shape:", df.shape)

print("\nColumn names:")
print(df.columns)

# Function to prepare data for mplfinance
def prepare_stock_data(df, stock):
    required_columns = [f'{stock}_Open', f'{stock}_High', f'{stock}_Low', f'{stock}_Close', f'{stock}_Volume']
    if all(col in df.columns for col in required_columns):
        stock_data = df[required_columns].copy()
        stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return stock_data
    else:
        print(f"Warning: Not all required columns present for {stock}. Skipping this stock.")
        return None

# Function to plot candlestick chart with volume
def plot_stock_chart(data, title):
    if data is not None and not data.empty:
        mpf.plot(data, type='candle', volume=True, title=title, style='yahoo',
                 figsize=(12, 8), panel_ratios=(2, 1))
    else:
        print(f"Warning: No data to plot for {title}")

# Plot charts for all stocks
stocks = list(set([col.split('_')[0] for col in df.columns if '_Close' in col]))

for stock in stocks:
    stock_data = prepare_stock_data(df, stock)
    plot_stock_chart(stock_data, f'{stock} Stock Price and Volume')

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())

# Display correlation matrix
print("\nCorrelation matrix:")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Plot the percentage of missing values for each column
missing_percentages = df.isnull().mean() * 100
plt.figure(figsize=(12, 6))
missing_percentages.plot(kind='bar')
plt.title('Percentage of Missing Values by Column')
plt.xlabel('Columns')
plt.ylabel('Percentage of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()