from prophet import Prophet

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def train(self, data):
        # Assuming data has 'ds' and 'y' columns
        self.model.fit(data)

    def predict(self, future_dates):
        future = self.model.make_future_dataframe(periods=len(future_dates))
        forecast = self.model.predict(future)
        return forecast['yhat']

