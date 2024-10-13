from src.imports import np, pd, plt
from src.models.xgboost_model import XGBoostModel, create_features
from src.utils import load_data
from src.config import DATA_PATH, TARGET_COLUMN, SPLIT_DATE
from sklearn.metrics import mean_squared_error
import os

def save_plot(fig, filename):
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)

def evaluate_forecast_horizons(model, data, max_days=14):
    errors = []
    for days in range(1, max_days + 1):
        test_data = data.iloc[-24*days:]
        features = create_features(test_data.drop(TARGET_COLUMN, axis=1))
        predictions = model.predict_multi_step(features, steps=24*days)
        mse = mean_squared_error(test_data[TARGET_COLUMN], predictions)
        errors.append((days, mse))
    
    days, mses = zip(*errors)
    best_horizon = days[np.argmin(mses)]
    second_best_horizon = days[sorted(range(len(mses)), key=lambda i: mses[i])[1]]
    print(f"Best forecast horizon: {best_horizon} days")
    print(f"Second best forecast horizon: {second_best_horizon} days")
    return days, mses, best_horizon, second_best_horizon

def main():
    # Load data
    data = load_data(DATA_PATH)
    data = data.sort_index()

    # Create features for the entire dataset
    features_df = create_features(data)
    features_df[TARGET_COLUMN] = data[TARGET_COLUMN]

    # Set aside last 14 days for final testing
    last_two_weeks = features_df.iloc[-336:]  # Last 14 days (14 * 24 hours)
    data_for_training = features_df.iloc[:-336]

    # Convert SPLIT_DATE to datetime
    split_date = pd.to_datetime(SPLIT_DATE)

    # Initialize and train the model
    model = XGBoostModel()
    model.fit(data_for_training, split_date, TARGET_COLUMN)

    # Plot 1: Train/Test Error During Training
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.model.evals_result()['validation_0']['rmse'], label='Train')
    ax.plot(model.model.evals_result()['validation_1']['rmse'], label='Test')
    ax.set_title('Train/Test Error During Training')
    ax.set_xlabel('Boosting Rounds')
    ax.set_ylabel('RMSE')
    ax.legend()
    save_plot(fig, 'train_test_error.png')

    # Evaluate the model and find the best forecast horizons
    days, mses, best_horizon, second_best_horizon = evaluate_forecast_horizons(model, last_two_weeks)

    # Plot 2: Forecast Error by Horizon
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, mses)
    ax.set_title('Forecast Error by Horizon')
    ax.set_xlabel('Forecast Horizon (Days)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xticks(days)
    ax.set_xticklabels(days, rotation=45)
    save_plot(fig, 'forecast_error_by_horizon.png')

    # Plot 3: Train/Predict Comparison for All Data
    train_features = data_for_training.drop(TARGET_COLUMN, axis=1)
    train_predictions = model.predict(train_features)
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(data_for_training.index, data_for_training[TARGET_COLUMN], label='Actual')
    ax.plot(data_for_training.index, train_predictions, label='Predicted')
    ax.set_title('Train/Predict Comparison (All Data)')
    ax.set_xlabel('Date')
    ax.set_ylabel(TARGET_COLUMN)
    ax.legend()
    save_plot(fig, 'train_predict_comparison.png')

    # Plots 4 and 5: Best and Second Best Horizon Forecasts
    for horizon in [best_horizon, second_best_horizon]:
        horizon_hours = horizon * 24
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=horizon_hours, freq='H')
        future_df = pd.DataFrame(index=future_dates)
        future_features = create_features(future_df)
        
        # Multi-step forecast
        multi_step_predictions = model.predict_multi_step(future_features, steps=horizon_hours)
        
        # Auto-regressive forecast
        auto_regressive_predictions = []
        current_features = future_features.iloc[0].to_dict()
        for _ in range(horizon_hours):
            pred = model.predict(pd.DataFrame([current_features]))[0]
            auto_regressive_predictions.append(pred)
            current_features['hour'] = (current_features['hour'] + 1) % 24
            if current_features['hour'] == 0:
                current_features['dayofweek'] = (current_features['dayofweek'] + 1) % 7
                current_features['dayofyear'] = current_features['dayofyear'] % 365 + 1
                current_features['dayofmonth'] = (current_features['dayofmonth'] % 28) + 1
                current_features['weekofyear'] = (current_features['weekofyear'] % 52) + 1
            if current_features['hour'] == 0 and current_features['dayofmonth'] == 1:
                current_features['month'] = (current_features['month'] % 12) + 1
                if current_features['month'] == 1:
                    current_features['year'] += 1
                    current_features['quarter'] = 1
                elif current_features['month'] in [4, 7, 10]:
                    current_features['quarter'] += 1

        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(last_two_weeks.index, last_two_weeks[TARGET_COLUMN], label='Actual')
        ax.plot(future_dates, multi_step_predictions, label=f'Multi-step Predictions ({horizon} days)', color='red')
        
        # Only plot auto-regressive predictions if they're different from multi-step
        if not np.allclose(multi_step_predictions, auto_regressive_predictions):
            ax.plot(future_dates, auto_regressive_predictions, label=f'Auto-regressive Predictions ({horizon} days)', color='green')
        
        ax.set_title(f'{horizon}-day Forecast ({"Best" if horizon == best_horizon else "Second Best"} Horizon)')
        ax.set_xlabel('Date')
        ax.set_ylabel(TARGET_COLUMN)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        save_plot(fig, f'{horizon}_day_forecast.png')

    # Optional: Save the model
    model.save_model('xgboost_full_model')

if __name__ == "__main__":
    main()
