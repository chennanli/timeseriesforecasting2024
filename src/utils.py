import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from src.config import PLOTS_DIR

def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Add other common functions here

# Add any utility functions you need here
def some_utility_function():
    pass  # Replace with actual implementation if needed

def save_plot(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close(fig)
