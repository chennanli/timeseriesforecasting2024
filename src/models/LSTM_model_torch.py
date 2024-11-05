import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from tqdm import tqdm
import time
from ..config import PLOTS_DIR
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out shape: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

class LSTMModelTorch:
    def __init__(self, sequence_length=20, forecast_horizon=1):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, df, target_column):
        scaled_data = self.scaler.fit_transform(df[[target_column]])
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data) - self.forecast_horizon + 1):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i:i+self.forecast_horizon, 0])
        return np.array(X), np.array(y)

    def fit(self, df, target_column, split_date, epochs=20, batch_size=32):
        df_train = df.loc[df.index <= split_date].copy()
        X_train, y_train = self.prepare_data(df_train, target_column)
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = LSTMModel(input_size=1, hidden_size=50, num_layers=1, output_size=self.forecast_horizon).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        history = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.unsqueeze(2).to(self.device)  # Shape: (batch_size, sequence_length, 1)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            history.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        return history

    def predict(self, df, target_column):
        self.model.eval()
        with torch.no_grad():
            scaled_data = self.scaler.transform(df[[target_column]])
            X = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                X.append(scaled_data[i:i+self.sequence_length, 0])
            X = np.array(X)
            X = torch.FloatTensor(X).unsqueeze(2).to(self.device)  # Shape: (num_samples, sequence_length, 1)
            
            predictions = self.model(X)
            predictions = self.scaler.inverse_transform(predictions.cpu().numpy())
            
        result_df = pd.DataFrame(predictions, index=df.index[self.sequence_length-1:], 
                                 columns=[f'LSTM_Prediction_t+{i+1}' for i in range(self.forecast_horizon)])
        return result_df

    def evaluate(self, df, target_column):
        y_true = df[target_column].values
        predictions = self.predict(df, target_column)
        y_pred = predictions['LSTM_Prediction'].values

        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: No data available for evaluation.")
            return {
                'MSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'RMSE': np.nan
            }

        mse = mean_squared_error(y_true[:len(y_pred)], y_pred)
        mae = mean_absolute_error(y_true[:len(y_pred)], y_pred)
        mape = mean_absolute_percentage_error(y_true[:len(y_pred)], y_pred)
        rmse = np.sqrt(mse)

        return {
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse
        }

def determine_sequence_length(df, target_column, method='autocorrelation'):
    # Implementation remains the same as in the TensorFlow version
    if method == 'autocorrelation':
        acf_values = acf(df[target_column], nlags=100)
        # Find the first lag where autocorrelation drops below 0.2
        sequence_length = next((i for i, x in enumerate(acf_values) if x < 0.2), 20)
    elif method == 'domain_expert':
        sequence_length = int(input("Enter the sequence length based on domain knowledge: "))
    else:
        raise ValueError("Invalid method. Choose 'autocorrelation' or 'domain_expert'.")
    
    return sequence_length

def run_lstm_model_torch(df, split_date, target_column, sequence_length_method='autocorrelation'):
    sequence_length = determine_sequence_length(df, target_column, method=sequence_length_method)
    print(f"Using sequence length: {sequence_length}")

    forecast_horizons = [1, 7, 14]  # Example values
    best_model = None
    best_rmse = float('inf')
    best_config = None
    best_history = None
    best_predictions = None

    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    for horizon in forecast_horizons:
        print(f"\nTraining model for forecast horizon: {horizon}")
        model = LSTMModelTorch(sequence_length=sequence_length, forecast_horizon=horizon)
        history = model.fit(df_train, target_column, split_date, epochs=20)
        
        predictions = model.predict(df_test, target_column)
        
        # Evaluate only on the first prediction for each time step
        metrics = model.evaluate(df_test.iloc[sequence_length-1:], target_column)
        print(f"Metrics for horizon {horizon}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        if metrics['RMSE'] < best_rmse:
            best_rmse = metrics['RMSE']
            best_model = model
            best_config = (sequence_length, horizon)
            best_history = history
            best_predictions = predictions

    print(f"\nBest configuration: sequence_length={best_config[0]}, forecast_horizon={best_config[1]}")
    return best_model, best_config[1], metrics, df_test.join(best_predictions), best_history

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history, label='Training Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'lstm_torch_training_history.png'))
    plt.close()
