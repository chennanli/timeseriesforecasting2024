# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Time series specific
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Other utilities
import os
import logging

# Project-specific imports
from .models.neural_forecast_model import NeuralForecastModel
from .models.arima_model import ARIMAModel

try:
    from .models.prophet_model import ProphetModel
except ImportError:
    print("Warning: ProphetModel could not be imported. Make sure the prophet package is installed.")

# You can add more imports as needed
from . import utils
from .utils import some_utility_function
