from src.common import load_data, calculate_metrics
from src.models.neural_forecast_model import NeuralForecastModel
from src.models.xgboost_model import run_xgboost_model
# Import other models as needed
# from src.models.arima_model import run_arima_model
# from src.models.prophet_model import run_prophet_model

def load_preprocessed_data(file_path):
    # Implement this function to load the preprocessed data
    pass

def run_selected_model(model_name, df, split_date, target_column):
    if model_name == 'neural_forecast':
        model = NeuralForecastModel()
        # Implement the fit and predict methods for NeuralForecastModel
        # Return model, metrics, and predictions
    elif model_name == 'xgboost':
        return run_xgboost_model(df, split_date, target_column)
    # Add other model options here
    # elif model_name == 'arima':
    #     return run_arima_model(df, split_date, target_column)
    # elif model_name == 'prophet':
    #     return run_prophet_model(df, split_date, target_column)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    # Clean and preprocess data
    data_cleanup("data/raw/input_data.csv", "data/processed/preprocessed_data.csv")

    # Load preprocessed data
    df = load_preprocessed_data("data/processed/preprocessed_data.csv")

    # Set your split date and target column
    split_date = '01-Jan-2015'
    target_column = 'PJME_MW'

    # Select the model you want to run
    selected_model = 'xgboost'  # Change this to run different models

    # Run the selected model
    model, metrics, predictions = run_selected_model(selected_model, df, split_date, target_column)

    # Print metrics
    print(metrics)

    # Do something with the predictions
    print(predictions.head())

if __name__ == "__main__":
    main()