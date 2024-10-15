import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.models.xgboost_model import XGBoostModel, create_features
from src.models.prophet_model import ProphetModel
import numpy as np
from datetime import timedelta
import os

# Load data
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "raw", "PJME_hourly.csv")
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

df = load_data()

# Forecasting functions
def forecast_xgboost(train_data, test_data, forecast_days, progress_bar, status_text):
    model = XGBoostModel()
    status_text.text("XGBoost: Fitting model...")
    model.fit(train_data, 'PJME_MW')
    progress_bar.progress(0.5)
    
    status_text.text("XGBoost: Making predictions...")
    X_test = create_features(test_data)
    predictions = pd.Series(model.predict(X_test), index=test_data.index)
    
    future_dates = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=forecast_days*24, freq='H')
    future_df = pd.DataFrame(index=future_dates)
    future_features = create_features(future_df)
    future_predictions = model.predict(future_features)
    progress_bar.progress(0.8)
    
    status_text.text("XGBoost: Calculating metrics...")
    metrics = model.evaluate(X_test, test_data['PJME_MW'])
    progress_bar.progress(1.0)
    status_text.text("XGBoost: Complete!")
    return predictions, future_predictions, metrics

def forecast_prophet(train_data, test_data, forecast_days, progress_bar, status_text):
    model = ProphetModel()
    status_text.text("Prophet: Fitting model...")
    model.fit(train_data, 'PJME_MW')
    progress_bar.progress(0.5)
    
    status_text.text("Prophet: Making predictions...")
    forecast = model.predict(test_data.index)
    predictions = pd.Series(forecast['yhat'].values, index=test_data.index)
    
    future_dates = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=forecast_days*24, freq='H')
    future_predictions = model.predict(future_dates)['yhat'].values
    progress_bar.progress(0.8)
    
    status_text.text("Prophet: Calculating metrics...")
    metrics = model.evaluate(test_data, 'PJME_MW')
    progress_bar.progress(1.0)
    status_text.text("Prophet: Complete!")
    return predictions, future_predictions, metrics

# Streamlit app
st.title('Time Series Forecasting App')

# Sidebar
st.sidebar.header('Settings')
use_xgboost = st.sidebar.checkbox('Use XGBoost', value=True)
use_prophet = st.sidebar.checkbox('Use Prophet', value=True)
forecast_days = st.sidebar.slider('Forecast days', 1, 30, 7)

# Split data
train_data = df[:'2014-12-31']
test_data = df['2015-01-01':]

# Run Model button
if st.sidebar.button('Run Model'):
    results = {}
    
    if use_xgboost:
        with st.expander("XGBoost Progress", expanded=True):
            xgb_progress = st.progress(0)
            xgb_status = st.empty()
            xgb_predictions, xgb_future_predictions, xgb_metrics = forecast_xgboost(train_data, test_data, forecast_days, xgb_progress, xgb_status)
            results['XGBoost'] = (xgb_predictions, xgb_future_predictions, xgb_metrics)
    
    if use_prophet:
        with st.expander("Prophet Progress", expanded=True):
            prophet_progress = st.progress(0)
            prophet_status = st.empty()
            prophet_predictions, prophet_future_predictions, prophet_metrics = forecast_prophet(train_data, test_data, forecast_days, prophet_progress, prophet_status)
            results['Prophet'] = (prophet_predictions, prophet_future_predictions, prophet_metrics)

    # Plot results
    st.subheader('Forecast Results')
    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['PJME_MW'], mode='lines', name='Actual'))

    # Plot predictions for each model
    colors = {'XGBoost': 'red', 'Prophet': 'green'}
    for model, (predictions, future_predictions, _) in results.items():
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name=f'{model} Predicted', line=dict(color=colors[model])))
        future_dates = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=forecast_days*24, freq='H')
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name=f'{model} Future Forecast', line=dict(color=colors[model], dash='dash')))

    fig.update_layout(title='Forecast Results', xaxis_title='Date', yaxis_title='PJME_MW', legend_title='Legend')
    st.plotly_chart(fig, use_container_width=True)

    # Display metrics in a single table
    st.subheader('Model Performance')
    metrics_df = pd.DataFrame({model: metrics for model, (_, _, metrics) in results.items()}).T
    st.table(metrics_df.style.format("{:.2f}"))

    # Display raw data
    st.subheader('Raw Data')
    st.write(df)
else:
    st.write("Please select the models you want to use and click 'Run Model' to see the results.")
