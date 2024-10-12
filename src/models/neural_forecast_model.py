from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

class NeuralForecastModel:
    def __init__(self, horizon):
        self.horizon = horizon
        self.model = None

    def train(self, df):
        self.model = NeuralForecast(
            models=[NHITS(input_size=2*self.horizon, h=self.horizon)],
            freq='D'
        )
        self.model.fit(df)

    def predict(self, df):
        return self.model.predict(df)

    def save_model(self, path):
        # Implement model saving logic
        pass

    def load_model(self, path):
        # Implement model loading logic
        pass
