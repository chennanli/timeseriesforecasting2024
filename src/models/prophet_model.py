import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def prepare_data(self, df, target_column):
        df_prophet = df.reset_index().rename(columns={'Datetime': 'ds', target_column: 'y'})
        return df_prophet

    def fit(self, df, target_column):
        df_prophet = self.prepare_data(df, target_column)
        self.model.fit(df_prophet)
        return self

    def predict(self, future_dates):
        future = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future)
        return forecast

    def evaluate(self, df, target_column):
        df_prophet = self.prepare_data(df, target_column)
        forecast = self.model.predict(df_prophet)
        y_true = df_prophet['y']
        y_pred = forecast['yhat']
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }

    def plot_components(self, forecast):
        return self.model.plot_components(forecast)

def run_prophet_model(df, split_date, target_column):
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    model = ProphetModel()
    model.fit(df_train, target_column)
    
    future_dates = df_test.index
    forecast = model.predict(future_dates)
    
    metrics = model.evaluate(df_test, target_column)
    
    return model, forecast, metrics, df_test
