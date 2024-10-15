import pandas as pd
import matplotlib.pyplot as plt
from src.models.xgboost_model import XGBoostModel, create_features
from src.models.prophet_model import ProphetModel
from src.models.TCN_model import TCNModel
from datetime import timedelta
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from itertools import product

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

def plot_results(actual, predictions, model_name, start_date, end_date):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.loc[start_date:end_date].index, actual.loc[start_date:end_date], label='Actual', alpha=0.5)
    plt.plot(predictions.loc[start_date:end_date].index, predictions.loc[start_date:end_date], label='Predicted', alpha=0.5)
    plt.title(f'{model_name} Model: Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def calculate_metrics(actual, predictions):
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def select_models():
    print("Select models to run:")
    print("1. XGBoost")
    print("2. Prophet")
    print("3. TCN")
    print("4. TCN with hyperparameter tuning")
    print("5. All models")
    choice = input("Enter the numbers of the models you want to run (comma-separated): ")
    choices = [int(c.strip()) for c in choice.split(',')]
    if 5 in choices:
        return ['XGBoost', 'Prophet', 'TCN', 'TCN_tuning']
    selected_models = []
    if 1 in choices:
        selected_models.append('XGBoost')
    if 2 in choices:
        selected_models.append('Prophet')
    if 3 in choices:
        selected_models.append('TCN')
    if 4 in choices:
        selected_models.append('TCN_tuning')
    return selected_models

def tcn_grid_search(X_train, y_train, X_test, y_test, sequence_length, forecast_horizon):
    param_grid = {
        'num_filters': [32, 64, 128],
        'kernel_size': [3, 5, 7],
        'dilations': [[1, 2, 4, 8], [1, 2, 4, 8, 16]],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001]
    }

    best_model = None
    best_rmse = float('inf')

    for params in product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))
        model = TCNModel(sequence_length, forecast_horizon, **current_params)
        model.build_model()
        model.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions.flatten()))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            
        print(f"Parameters: {current_params}")
        print(f"RMSE: {rmse}")
        print("--------------------")

    return best_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "raw", "PJME_hourly.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    target_column = "PJME_MW"
    df = load_data(file_path)
    
    # Use the last 20% of the data for testing
    split_date = df.index[-int(len(df)*0.2)]
    train_data = df.loc[:split_date].copy()
    test_data = df.loc[split_date:].copy()

    # Increased sequence length for TCN
    sequence_length = 168  # One week of hourly data
    forecast_horizon = 24

    selected_models = select_models()

    models = {
        'XGBoost': XGBoostModel(),
        'Prophet': ProphetModel(),
        'TCN': TCNModel(sequence_length, forecast_horizon)
    }

    results = {}
    predictions = {}

    for name in selected_models:
        print(f"\nTraining {name} model...")
        
        if name == 'XGBoost':
            model = models[name]
            model.fit(train_data, target_column)
            X_test = create_features(test_data)
            predictions[name] = pd.Series(model.predict(X_test), index=test_data.index)
            results[name] = model.evaluate(X_test, test_data[target_column])
        elif name == 'Prophet':
            model = models[name]
            model.fit(train_data, target_column)
            forecast = model.predict(test_data.index)
            predictions[name] = pd.Series(forecast['yhat'].values, index=test_data.index)
            results[name] = model.evaluate(test_data, target_column)
        elif name == 'TCN':
            model = models[name]
            X_train, X_test, y_train, y_test, test_index = model.load_and_preprocess_data(file_path, target_column)
            model.build_model()
            history = model.train(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
            tcn_predictions = model.predict(X_test)
            predictions[name] = pd.Series(tcn_predictions.flatten(), index=test_index[:len(tcn_predictions.flatten())])
            results[name] = calculate_metrics(y_test.flatten(), tcn_predictions.flatten())
            
            # Plot training history for TCN
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('TCN Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        elif name == 'TCN_tuning':
            print("Performing TCN hyperparameter tuning...")
            model = TCNModel(sequence_length, forecast_horizon)
            X_train, X_test, y_train, y_test, test_index = model.load_and_preprocess_data(file_path, target_column)
            best_model = tcn_grid_search(X_train, y_train, X_test, y_test, sequence_length, forecast_horizon)
            tcn_predictions = best_model.predict(X_test)
            predictions[name] = pd.Series(tcn_predictions.flatten(), index=test_index[:len(tcn_predictions.flatten())])
            results[name] = calculate_metrics(y_test.flatten(), tcn_predictions.flatten())

        if name != 'TCN_tuning':
            plot_results(test_data[target_column], predictions[name], name, test_data.index[0], test_data.index[-1])
        print(f"{name} Metrics:", results[name])

    # Compare results
    if len(selected_models) > 1:
        print("\nComparison of Models:")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df)

        # Plot future predictions
        future_days = 7
        future_dates = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=future_days*24, freq='H')
        
        plt.figure(figsize=(12, 6))
        for name in selected_models:
            if name == 'XGBoost':
                model = models[name]
                future_df = pd.DataFrame(index=future_dates)
                future_features = create_features(future_df)
                future_predictions = model.predict(future_features)
            elif name == 'Prophet':
                model = models[name]
                future_predictions = model.predict(future_dates)['yhat'].values
            elif name == 'TCN':
                model = models[name]
                initial_sequence = X_test[-1].reshape(1, sequence_length, 1)
                future_predictions = model.predict_sliding_window(initial_sequence, future_days*24).flatten()
            elif name == 'TCN_tuning':
                initial_sequence = X_test[-1].reshape(1, sequence_length, 1)
                future_predictions = best_model.predict_sliding_window(initial_sequence, future_days*24).flatten()

            plt.plot(future_dates, future_predictions, label=name, alpha=0.7)

        plt.title('Future Predictions Comparison')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    print("\nForecasting complete. Check the plots for results.")

if __name__ == "__main__":
    main()
