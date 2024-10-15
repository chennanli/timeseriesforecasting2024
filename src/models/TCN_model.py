import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, BatchNormalization, Activation, Add, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

class WeightNormalization(Callback):
    def __init__(self):
        self.layer_to_weights = {}

    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, Conv1D):
                weights = layer.get_weights()
                self.layer_to_weights[layer] = weights

    def on_batch_end(self, batch, logs=None):
        for layer, original_weights in self.layer_to_weights.items():
            weights = layer.get_weights()
            weights[0] = original_weights[0] * K.l2_normalize(weights[0], axis=(0, 1))
            layer.set_weights(weights)

class TCNModel:
    def __init__(self, sequence_length, forecast_horizon, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.2, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
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
        return X_train, X_test, y_train, y_test, df.index[-len(X_test):]

    def residual_block(self, x, dilation_rate, nb_filters, kernel_size):
        prev_x = x
        x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        if prev_x.shape[-1] != nb_filters:
            prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
        
        res = Add()([prev_x, x])
        return res

    def build_model(self):
        inputs = Input(shape=(self.sequence_length, 1))
        x = inputs

        for dilation in self.dilations:
            x = self.residual_block(x, dilation, self.num_filters, self.kernel_size)

        x = Conv1D(filters=self.num_filters, kernel_size=1, activation='relu')(x)
        x = Lambda(lambda y: y[:, -1, :])(x)
        outputs = Dense(self.forecast_horizon)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        weight_norm = WeightNormalization()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr, weight_norm],
            verbose=1
        )
        return history

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions)

    def predict_sliding_window(self, initial_sequence, num_predictions):
        predictions = []
        current_sequence = initial_sequence.copy()

        for _ in range(num_predictions):
            next_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1))
            predictions.append(next_pred[0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]

        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.flatten(), label='Actual')
        plt.plot(y_pred.flatten(), label='Predicted')
        plt.title('TCN Model: Actual vs Predicted')
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
