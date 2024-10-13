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
    df['weekofyear'] = df.index.isocalendar().week

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
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
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit(self, df, split_date, target_column):
        df = df.sort_index()
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
            verbose=100
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

    def predict_multi_step(self, X, steps):
        predictions = []
        current_features = X.iloc[0].to_dict()
        for _ in range(steps):
            pred = self.model.predict(pd.DataFrame([current_features]))[0]
            predictions.append(pred)
            # Update time-based features for next prediction
            current_features['hour'] = (current_features['hour'] + 1) % 24
            if current_features['hour'] == 0:
                current_features['dayofweek'] = (current_features['dayofweek'] + 1) % 7
                current_features['dayofyear'] = current_features['dayofyear'] % 365 + 1
                current_features['dayofmonth'] = (current_features['dayofmonth'] % 28) + 1  # Simplified
                current_features['weekofyear'] = (current_features['weekofyear'] % 52) + 1
            if current_features['hour'] == 0 and current_features['dayofmonth'] == 1:
                current_features['month'] = (current_features['month'] % 12) + 1
                if current_features['month'] == 1:
                    current_features['year'] += 1
                    current_features['quarter'] = 1
                elif current_features['month'] in [4, 7, 10]:
                    current_features['quarter'] += 1
        return predictions

def run_xgboost_model(df, split_date, target_column):
    model = XGBoostModel()
    model.fit(df, split_date, target_column)
    metrics = model.evaluate()

    df_test = df.loc[df.index > split_date].copy()
    X_test = create_features(df_test)
    df_test['MW_Prediction'] = model.predict(X_test)

    return model, metrics, df_test
