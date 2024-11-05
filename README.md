# Energy Consumption Forecasting

This project focuses on forecasting energy consumption using various time series models.

## Models Implemented

- XGBoost
- Prophet

## Project Structure

```
TimeSeriesForecasting_Sep2024/
├── data/
│   ├── raw/             # Raw energy consumption data
│   └── processed/       # Processed datasets
├── notebooks/          # Jupyter notebooks for analysis
├── src/
│   ├── models/         # Model implementations
│   └── utils/          # Utility functions
├── main.py            # Main script for running predictions
└── streamlit_app.py   # Interactive web application
```

## Getting Started

1. Ensure the energy consumption data files are in the `data/raw/` directory
2. Install required dependencies
3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Data Files

The project uses hourly energy consumption data from various regions:
- PJME_hourly.csv
- AEP_hourly.csv
- COMED_hourly.csv
- etc.
