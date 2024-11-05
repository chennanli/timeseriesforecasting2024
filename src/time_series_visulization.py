import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('fivethirtyeight')

def plot_time_series(df):
    """
    Create time series plots for all numeric columns in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time series data with datetime index
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing all the plots
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    n_features = len(numeric_df.columns)
    
    if n_features == 0:
        raise ValueError("No numeric columns found in the dataframe")
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 5*n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    
    # Find global min and max for y-axis
    y_min = numeric_df.min().min()
    y_max = numeric_df.max().max()
    
    # Plot each feature
    for i, column in enumerate(numeric_df.columns):
        axes[i].plot(numeric_df.index, numeric_df[column], '.-', label=column)
        axes[i].set_title(f'Time Series: {column}')
        axes[i].set_ylabel('Value')
        axes[i].set_ylim(y_min, y_max)
        axes[i].legend()
    
    # Set x-axis label only for the bottom subplot
    axes[-1].set_xlabel('Date')
    
    plt.tight_layout()
    return fig

def load_and_visualize_data(data_path=None):
    """
    Main function to load CSV files, create a summary, and visualize time series data.
    Can be used both standalone and from Streamlit.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data directory. If None, uses the default project structure.
    """
    # Get the project root directory if no path provided
    if data_path is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(PROJECT_ROOT, 'data', 'raw')
    
    # Get list of all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    
    # Skip specific files when running standalone
    files_to_skip = ['hrl_load_metered.csv']  # Add more files to skip if needed
    if data_path.endswith('raw'):  # Only skip files in default mode
        csv_files = [f for f in csv_files if f not in files_to_skip]

    # Dictionary to store all dataframes
    dfs = {}

    # Load each CSV file
    for file in csv_files:
        df_name = file.replace('.csv', '').lower()
        try:
            df = pd.read_csv(
                os.path.join(data_path, file),
                parse_dates=True,
                index_col=0
            )
            dfs[df_name] = df
            print(f"Successfully loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue

    if not dfs:
        print("No CSV files were successfully loaded.")
        return None, {}

    # Create a summary table
    summary_data = []
    for name, df in dfs.items():
        summary_data.append({
            'File Name': name,
            'Start Time': df.index.min(),
            'End Time': df.index.max(),
            'Number of Features': len(df.columns),
            'Feature Names': ', '.join(df.columns),
            'DataFrame Shape': f"{df.shape[0]} rows x {df.shape[1]} columns"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nSummary of CSV Files:")
    print(summary_df.to_string(index=False))
    print("\n")

    # When running standalone, automatically plot all files
    if data_path.endswith('raw'):
        for name, df in dfs.items():
            print(f"\nPlotting time series for file: {name}")
            try:
                fig = plot_time_series(df)
                plt.show()
                plt.close()
            except Exception as e:
                print(f"Error plotting {name}: {str(e)}")

    return summary_df, dfs

if __name__ == "__main__":
    # When run as standalone script, use default project structure
    load_and_visualize_data()
