import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Core paths (automatic/reliable)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Environment-specific settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
API_KEY = os.getenv('API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

# Project paths
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

# Data files
DATA_FILE = 'consolidated_data.csv'  # Added for main.py
DATA_FILE_NAME = 'consolidated_data.csv'  # Kept for backward compatibility
CONSOLIDATED_DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
TICKER_RAW_FOLDER = os.path.join(RAW_DATA_DIR, 'ticker_raw')

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configuration - Time Period Settings
TEST_PERIOD_WEEKS = 2
VALIDATION_PERIOD_WEEKS = 4
BUFFER_DAYS = 1
MAX_DAYS_MISSING = 30

# Convert user settings to timedelta objects
TEST_PERIOD = timedelta(weeks=TEST_PERIOD_WEEKS)
VALIDATION_PERIOD = timedelta(weeks=VALIDATION_PERIOD_WEEKS)
BUFFER_PERIOD = timedelta(days=BUFFER_DAYS)

# Target column configuration
TARGET_COLUMN = 'Close'  # Default target column
DATE_COLUMN = 'datetime'  # Default date column

# Split date (can be overridden)
SPLIT_DATE = '2015-01-01'  # Added for main.py

def get_split_dates():
    """
    Splits the data into train/validation/test periods based on user-defined settings.
    
    Returns:
    --------
    dict
        Dictionary containing split dates for train, validation, and test sets
    """
    today = datetime.now().date()
    test_start = today - TEST_PERIOD
    validation_start = test_start - VALIDATION_PERIOD
    return {
        'train_end': validation_start - BUFFER_PERIOD,
        'validation_start': validation_start,
        'validation_end': test_start - BUFFER_PERIOD,
        'test_start': test_start,
        'test_end': today
    }

# List of all available tickers (to be populated dynamically)
AVAILABLE_TICKERS = []
