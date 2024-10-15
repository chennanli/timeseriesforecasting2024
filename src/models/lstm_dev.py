import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class LSTMModel:
    def __init__(self, sequence_length, forecast_horizon):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self, file_path, target_column):
        df = pd.read_csv(file_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

        data = self.scaler.fit_transform(df[[target_column]])

        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon)])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        self.model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(self.forecast_horizon)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions)

    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('LSTM Model: Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def plot_history(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def main():
    print("Current working directory:", os.getcwd())

    file_path = os.path.join("TimeSeriesForecasting_Sep2024", "data", "raw", "PJME_hourly.csv")
    
    full_path = os.path.abspath(file_path)
    print("Full path of the file:", full_path)

    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}")
        return

    target_column = "PJME_MW"
    sequence_length = 24  # 24 hours
    forecast_horizon = 24  # Predict next 24 hours

    lstm_model = LSTMModel(sequence_length, forecast_horizon)
    X_train, X_test, y_train, y_test = lstm_model.load_and_preprocess_data(full_path, target_column)

    lstm_model.build_model()
    history = lstm_model.train(X_train, y_train, epochs=10)

    predictions = lstm_model.predict(X_test)

    y_true = lstm_model.scaler.inverse_transform(y_test.reshape(-1, forecast_horizon))
    lstm_model.plot_results(y_true.flatten(), predictions.flatten())
    lstm_model.plot_history(history)

if __name__ == "__main__":
    main()
