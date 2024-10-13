from src.imports import np, pd, plt
from src.models.xgboost_model import XGBoostModel, create_features, run_xgboost_model
from src.models.prophet_model import ProphetModel, run_prophet_model
from src.utils import load_data
from src.config import DATA_PATH, TARGET_COLUMN, SPLIT_DATE
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os

def save_plot(fig, filename):
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, filename))
    plt.show()  # Add this line to display the plot
    plt.close(fig)

def main():
    # Load data
    data = load_data(DATA_PATH)
    data = data.sort_index()

    # Convert SPLIT_DATE to datetime
    split_date = pd.to_datetime(SPLIT_DATE)

    # Choose model(s) to run
    models_to_run = input("Enter models to run (xgboost, prophet, or both): ").lower()

    if 'xgboost' in models_to_run or 'both' in models_to_run:
        run_xgboost_analysis(data, split_date)

    if 'prophet' in models_to_run or 'both' in models_to_run:
        run_prophet_analysis(data, split_date)

    if 'both' in models_to_run:
        compare_models(data, split_date)

def run_xgboost_analysis(data, split_date):
    xgb_model, xgb_metrics, df_test = run_xgboost_model(data, split_date, TARGET_COLUMN)
    
    print("XGBoost Model Metrics:")
    for metric, value in xgb_metrics.items():
        print(f"{metric}: {value}")

    # Plot XGBoost results
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(df_test.index, df_test[TARGET_COLUMN], label='Actual')
    ax.plot(df_test.index, df_test['MW_Prediction'], label='XGBoost Prediction')
    ax.set_title('XGBoost Model: Actual vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel(TARGET_COLUMN)
    ax.legend()
    save_plot(fig, 'xgboost_prediction.png')

def run_prophet_analysis(data, split_date):
    prophet_model, prophet_forecast, prophet_metrics, df_test = run_prophet_model(data, split_date, TARGET_COLUMN)
    
    print("Prophet Model Metrics:")
    for metric, value in prophet_metrics.items():
        print(f"{metric}: {value}")

    # Plot Prophet results
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(df_test.index, df_test[TARGET_COLUMN], label='Actual')
    ax.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet Prediction')
    ax.fill_between(prophet_forecast['ds'], prophet_forecast['yhat_lower'], prophet_forecast['yhat_upper'], alpha=0.3)
    ax.set_title('Prophet Model: Actual vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel(TARGET_COLUMN)
    ax.legend()
    save_plot(fig, 'prophet_prediction.png')

    # Plot Prophet components
    components_fig = prophet_model.plot_components(prophet_forecast)
    save_plot(components_fig, 'prophet_components.png')

def compare_models(data, split_date):
    xgb_model, xgb_metrics, xgb_df_test = run_xgboost_model(data, split_date, TARGET_COLUMN)
    prophet_model, prophet_forecast, prophet_metrics, prophet_df_test = run_prophet_model(data, split_date, TARGET_COLUMN)

    # Combine predictions
    combined_df = xgb_df_test[[TARGET_COLUMN, 'MW_Prediction']].copy()
    combined_df = combined_df.rename(columns={'MW_Prediction': 'XGBoost_Prediction'})
    combined_df['Prophet_Prediction'] = prophet_forecast.set_index('ds')['yhat']

    # Forecast for 1 day and 7 days
    future_dates_1d = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
    future_dates_7d = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=24*7, freq='H')

    xgb_forecast_1d = xgb_model.predict_multi_step(create_features(pd.DataFrame(index=future_dates_1d)), steps=24)
    xgb_forecast_7d = xgb_model.predict_multi_step(create_features(pd.DataFrame(index=future_dates_7d)), steps=24*7)

    prophet_forecast_1d = prophet_model.predict(future_dates_1d)
    prophet_forecast_7d = prophet_model.predict(future_dates_7d)

    # Calculate metrics for different forecast windows
    def calculate_forecast_metrics(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)
        rmse = np.sqrt(mse)
        return {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}

    xgb_metrics_1d = calculate_forecast_metrics(combined_df[TARGET_COLUMN][-24:], xgb_forecast_1d)
    xgb_metrics_7d = calculate_forecast_metrics(combined_df[TARGET_COLUMN][-168:], xgb_forecast_7d)
    prophet_metrics_1d = calculate_forecast_metrics(combined_df[TARGET_COLUMN][-24:], prophet_forecast_1d['yhat'])
    prophet_metrics_7d = calculate_forecast_metrics(combined_df[TARGET_COLUMN][-168:], prophet_forecast_7d['yhat'])

    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))
    
    # Plot 1: Full time series comparison
    ax1.plot(combined_df.index, combined_df[TARGET_COLUMN], label='Actual', alpha=0.7)
    ax1.plot(combined_df.index, combined_df['XGBoost_Prediction'], label='XGBoost Prediction', alpha=0.7)
    ax1.plot(combined_df.index, combined_df['Prophet_Prediction'], label='Prophet Prediction', alpha=0.7)
    ax1.set_title('Model Comparison: XGBoost vs Prophet (Full Time Series)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(TARGET_COLUMN)
    ax1.legend()

    # Plot 2: 1-day forecast
    ax2.plot(combined_df.index[-24:], combined_df[TARGET_COLUMN][-24:], label='Actual', alpha=0.7)
    ax2.plot(future_dates_1d, xgb_forecast_1d, label='XGBoost Forecast', alpha=0.7)
    ax2.plot(future_dates_1d, prophet_forecast_1d['yhat'], label='Prophet Forecast', alpha=0.7)
    ax2.set_title('1-Day Forecast Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(TARGET_COLUMN)
    ax2.legend()

    # Plot 3: 7-day forecast
    ax3.plot(combined_df.index[-168:], combined_df[TARGET_COLUMN][-168:], label='Actual', alpha=0.7)
    ax3.plot(future_dates_7d, xgb_forecast_7d, label='XGBoost Forecast', alpha=0.7)
    ax3.plot(future_dates_7d, prophet_forecast_7d['yhat'], label='Prophet Forecast', alpha=0.7)
    ax3.set_title('7-Day Forecast Comparison')
    ax3.set_xlabel('Date')
    ax3.set_ylabel(TARGET_COLUMN)
    ax3.legend()

    plt.tight_layout()

    # Add a table with metrics
    metrics_data = [
        ['Metric', 'XGBoost (1d)', 'Prophet (1d)', 'XGBoost (7d)', 'Prophet (7d)'],
        ['RMSE', f"{xgb_metrics_1d['RMSE']:.2f}", f"{prophet_metrics_1d['RMSE']:.2f}", 
         f"{xgb_metrics_7d['RMSE']:.2f}", f"{prophet_metrics_7d['RMSE']:.2f}"],
        ['MAE', f"{xgb_metrics_1d['MAE']:.2f}", f"{prophet_metrics_1d['MAE']:.2f}", 
         f"{xgb_metrics_7d['MAE']:.2f}", f"{prophet_metrics_7d['MAE']:.2f}"],
        ['MAPE', f"{xgb_metrics_1d['MAPE']:.2f}%", f"{prophet_metrics_1d['MAPE']:.2f}%", 
         f"{xgb_metrics_7d['MAPE']:.2f}%", f"{prophet_metrics_7d['MAPE']:.2f}%"]
    ]

    table = ax3.table(cellText=metrics_data, loc='bottom', cellLoc='center', colWidths=[0.2]*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    save_plot(fig, 'model_comparison_with_forecasts.png')

    # Print metrics comparison
    print("\nMetrics Comparison:")
    print(f"{'Metric':<10} {'XGBoost (1d)':>15} {'Prophet (1d)':>15} {'XGBoost (7d)':>15} {'Prophet (7d)':>15}")
    print("-" * 80)
    for metric in ['RMSE', 'MAE', 'MAPE']:
        print(f"{metric:<10} {xgb_metrics_1d[metric]:>15.2f} {prophet_metrics_1d[metric]:>15.2f} "
              f"{xgb_metrics_7d[metric]:>15.2f} {prophet_metrics_7d[metric]:>15.2f}")

    # Determine the better model for each forecast window
    if xgb_metrics_1d['RMSE'] < prophet_metrics_1d['RMSE']:
        print("\nXGBoost model performs better for 1-day forecasts.")
    else:
        print("\nProphet model performs better for 1-day forecasts.")

    if xgb_metrics_7d['RMSE'] < prophet_metrics_7d['RMSE']:
        print("XGBoost model performs better for 7-day forecasts.")
    else:
        print("Prophet model performs better for 7-day forecasts.")

if __name__ == "__main__":
    main()