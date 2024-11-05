import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PROJECT_ROOT, 
    DATA_FILE_NAME as DATA_FILE, 
    PLOTS_DIR, 
    TARGET_COLUMN,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)
from src.utils import save_plot

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def ensure_data_path(data_file):
    """
    Ensure the data file exists and return its full path.
    First checks in the current directory, then in data/raw.
    """
    # Try current directory
    if os.path.exists(data_file):
        return data_file
    
    # Try processed directory
    processed_data_path = os.path.join(PROCESSED_DATA_DIR, data_file)
    if os.path.exists(processed_data_path):
        return processed_data_path
    
    # Try raw data directory
    raw_data_path = os.path.join(RAW_DATA_DIR, data_file)
    if os.path.exists(raw_data_path):
        return raw_data_path
    
    # Try project root
    project_data_path = os.path.join(PROJECT_ROOT, data_file)
    if os.path.exists(project_data_path):
        return project_data_path
    
    raise FileNotFoundError(f"Could not find {data_file} in any of the data directories")

# Rest of the main.py content remains the same...
