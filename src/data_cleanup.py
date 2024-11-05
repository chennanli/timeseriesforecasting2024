import os
import sys
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_FILE_NAME, MAX_DAYS_MISSING

def load_data():
    """Load data from the CSV file."""
    file_path = os.path.join(RAW_DATA_DIR, DATA_FILE_NAME)
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)  # Sort by datetime
    return df

def analyze_data(df):
    """Analyze the loaded data and print summary statistics."""
    print("DataFrame Summary:")
    print(df.info())
    
    print("\nPercentage of zeros in each feature:")
    zero_percentages = (df == 0).mean() * 100
    for column, percentage in zero_percentages.items():
        print(f"{column}: {percentage:.2f}%")
    
    print(f"\nStart date: {df['datetime'].min()}")
    print(f"End date: {df['datetime'].max()}")

def clean_data(df):
    """
    Clean the loaded data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # If DataFrame doesn't have an index name, assume it's datetime
    if df.index.name is None and isinstance(df.index, pd.DatetimeIndex):
        df.index.name = 'datetime'
    
    # If datetime is a column, set it as index
    elif 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    if constant_columns:
        df = df.drop(columns=constant_columns)
        print(f"Removed constant columns: {constant_columns}")
    
    return df

def preprocess_data(df):
    """Preprocess the cleaned data for modeling."""
    latest_date = df.index.max()
    cutoff_date = latest_date - timedelta(days=MAX_DAYS_MISSING)
    
    columns_to_drop = []
    for column in df.columns:
        if df[column].last_valid_index() < cutoff_date:
            columns_to_drop.append(column)
    
    df_preprocessed = df.drop(columns=columns_to_drop)
    
    dropped_percentage = (len(columns_to_drop) / len(df.columns)) * 100
    print(f"\n{dropped_percentage:.2f}% of features were dropped.")
    print("Dropped features:")
    print(", ".join(columns_to_drop))
    
    print("\nRemaining features summary:")
    summary_df = pd.DataFrame({
        'Feature': df_preprocessed.columns,
        'Start Date': [df_preprocessed[col].first_valid_index() for col in df_preprocessed.columns],
        'End Date': [df_preprocessed[col].last_valid_index() for col in df_preprocessed.columns]
    })
    print(summary_df.to_string())
    
    return df_preprocessed

def main():
    df = load_data()
    analyze_data(df)
    df_cleaned = clean_data(df)
    df_preprocessed = preprocess_data(df_cleaned)
    
    output_file = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_data.csv')
    df_preprocessed.to_csv(output_file)
    print(f"\nPreprocessed data saved to {output_file}")

if __name__ == "__main__":
    main()
