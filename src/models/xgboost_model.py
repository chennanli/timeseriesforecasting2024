import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class XGBoostModel:
    def __init__(self, n_estimators=1000, early_stopping_rounds=50):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, early_stopping_rounds=early_stopping_rounds)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit(self, df, split_date, target_column):
        self.prepare_data(df, split_date, target_column)
        self.train()
        return self

    def prepare_data(self, df, split_date, target_column):
        df_train = df.loc[df.index <= split_date].copy()
        df_test = df.loc[df.index > split_date].copy()

        self.X_train, self.y_train = create_features(df_train, label=target_column)
        self.X_test, self.y_test = create_features(df_test, label=target_column)

    def train(self):
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=False
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }

def run_xgboost_model(df, split_date, target_column):
    model = XGBoostModel()
    model.fit(df, split_date, target_column)
    metrics = model.evaluate()

    df_test = df.loc[df.index > split_date].copy()
    X_test = create_features(df_test)
    df_test['MW_Prediction'] = model.predict(X_test)

    return model, metrics, df_test