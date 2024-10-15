import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import os

class LSTMModel:
    def __init__(self, sequence_length=20, forecast_horizon=1):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df, target_column):
        scaled_data = self.scaler.fit_transform(df[[target_column]])
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data) - self.forecast_horizon + 1):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i:i+self.forecast_horizon, 0])
        return np.array(X), np.array(y)

    def create_model(self):
        model = Sequential()
        model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation="relu", return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.forecast_horizon))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, df, target_column, split_date, epochs=50, batch_size=32):
        df_train = df.loc[df.index <= split_date].copy()
        X_train, y_train = self.prepare_data(df_train, target_column)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        self.model = self.create_model()
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_split=0.2, verbose=1)
        return self, history

    def predict(self, df, target_column):
        # Use the last 'sequence_length' days of data for prediction
        last_known_values = df[target_column].values[-self.sequence_length:]
        
        if len(last_known_values) < self.sequence_length:
            print(f"Warning: Input has fewer rows ({len(last_known_values)}) than sequence length ({self.sequence_length}). Padding with the earliest available value.")
            pad_length = self.sequence_length - len(last_known_values)
            last_known_values = np.pad(last_known_values, (pad_length, 0), 'edge')
        
        scaled_input = self.scaler.transform(last_known_values.reshape(-1, 1))
        scaled_input = scaled_input.reshape(1, self.sequence_length, 1)
        
        scaled_prediction = self.model.predict(scaled_input)
        prediction = self.scaler.inverse_transform(scaled_prediction)
        
        # Create a DataFrame with predictions
        future_date = df.index[-1] + pd.Timedelta(days=1)
        result_df = pd.DataFrame(prediction, index=[future_date], columns=['LSTM_Prediction'])
        
        return result_df

    def evaluate(self, df, target_column):
        y_true = df[target_column].values[-self.forecast_horizon:]
        predictions = self.predict(df[:-self.forecast_horizon], target_column)
        y_pred = predictions['LSTM_Prediction'].values

        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: No data available for evaluation.")
            return {
                'MSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan,
                'RMSE': np.nan
            }

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

def determine_sequence_length(df, target_column, method='autocorrelation'):
    if method == 'autocorrelation':
        acf_values = acf(df[target_column], nlags=100)
        # Find the first lag where autocorrelation drops below 0.2
        sequence_length = next((i for i, x in enumerate(acf_values) if x < 0.2), 20)
    elif method == 'domain_expert':
        sequence_length = int(input("Enter the sequence length based on domain knowledge: "))
    else:
        raise ValueError("Invalid method. Choose 'autocorrelation' or 'domain_expert'.")
    
    return sequence_length

def run_lstm_model(df, split_date, target_column, sequence_length_method='autocorrelation'):
    sequence_length = determine_sequence_length(df, target_column, method=sequence_length_method)
    print(f"Using sequence length: {sequence_length}")

    forecast_horizon = 1  # Set to 1 as per your requirement
    model = LSTMModel(sequence_length=sequence_length, forecast_horizon=forecast_horizon)
    model, history = model.fit(df, target_column, split_date)

    df_test = df.loc[df.index > split_date].copy()
    
    # Ensure we have enough data for testing
    if len(df_test) < sequence_length:
        print(f"Warning: Test set is smaller than sequence length. Using last {sequence_length} rows of data for testing.")
        df_test = df.iloc[-sequence_length:].copy()

    predictions = model.predict(df_test, target_column)
    metrics = model.evaluate(df_test, target_column)

    print(f"Best configuration: sequence_length={sequence_length}, forecast_horizon={forecast_horizon}")
    return model, forecast_horizon, metrics, df_test.join(predictions), history

def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Training History')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    
    # Create 'plots' directory if it doesn't exist
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save the plot in the 'plots' directory
    plt.savefig(os.path.join(plots_dir, 'lstm_training_history.png'))
    return fig
