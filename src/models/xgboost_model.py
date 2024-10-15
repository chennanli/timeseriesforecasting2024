import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
from datetime import datetime
import matplotlib.pyplot as plt

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype('int32')

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    
    X = X.astype('int32')
    
    if label:
        y = df[label]
        return X, y
    return X

FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            early_stopping_rounds=50,
            learning_rate=0.01,
            max_depth=5
        )
        self.X = None
        self.y = None

    def fit(self, df, target_column):
        df = df.sort_index()
        self.prepare_data(df, target_column)
        self.train()
        return self

    def prepare_data(self, df, target_column):
        self.X, self.y = create_features(df, label=target_column)

    def train(self):
        self.model.fit(
            self.X, self.y,
            eval_set=[(self.X, self.y)],
            verbose=100
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }

    def predict_future(self, future_df):
        future_features = create_features(future_df)
        return self.model.predict(future_features)

    def save_model(self, custom_name=None):
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models', 'xgboost')
        os.makedirs(base_path, exist_ok=True)
        
        if custom_name:
            filename = f"{custom_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xgboost_model_{timestamp}.json"
        
        full_path = os.path.join(base_path, filename)
        self.model.save_model(full_path)
        print(f"Model saved to {full_path}")

    def load_model(self, filename):
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'saved_models', 'xgboost')
        full_path = os.path.join(base_path, filename)
        self.model.load_model(full_path)
        print(f"Model loaded from {full_path}")

    def create_features(self, df):
        return create_features(df)

def run_xgboost_model(df, split_date, target_column):
    model = XGBoostModel()
    model.fit(df, target_column)

    df_test = df.loc[df.index > split_date].copy()
    X_test, y_test = create_features(df_test, label=target_column)
    metrics = model.evaluate(X_test, y_test)
    df_test['MW_Prediction'] = model.predict(X_test)

    return model, metrics, df_test
