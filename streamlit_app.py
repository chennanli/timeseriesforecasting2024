import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import io
import plotly.graph_objects as go

# Local module imports
from src.time_series_visulization import plot_time_series, load_and_visualize_data
from src.data_cleanup import clean_data, analyze_data
from src.models.xgboost_model import XGBoostModel, create_features, run_xgboost_model
from src.models.prophet_model import ProphetModel

# Configure the Streamlit page
st.set_page_config(layout="wide", page_title="Time Series Analysis")

# Add custom CSS for bordered sections
st.markdown("""
<style>
.stBlock {
    padding: 20px;
    border-radius: 5px;
    border: 2px solid #f0f2f6;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Forecasting Models", "Time Series Visualization", "Data Cleanup"])

# Tab 1: Forecasting Models
with tab1:
    st.title('Time Series Forecasting App')

    # File Upload Section
    with st.container():
        st.markdown('<div class="stBlock">', unsafe_allow_html=True)
        st.subheader("1. Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file containing time series data with a 'Datetime' column",
            key="forecast_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Load the uploaded data
        df = pd.read_csv(uploaded_file)
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Target Selection Section
            st.markdown('<div class="stBlock">', unsafe_allow_html=True)
            st.subheader("2. Target Feature Selection")
            # Get available numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Show sample of data
            st.write("Preview of available features:")
            st.dataframe(df.head())
            
            # Target column selection with radio buttons
            target_col = st.radio(
                "Choose the feature to forecast:",
                options=numeric_columns,
                horizontal=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Model Selection Section
            st.markdown('<div class="stBlock">', unsafe_allow_html=True)
            st.subheader("3. Model Configuration")
            
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                use_xgboost = st.checkbox('Use XGBoost', value=True)
                use_prophet = st.checkbox('Use Prophet', value=True)
            
            # Forecast parameters
            with col2:
                forecast_days = st.slider('Forecast Horizon (days)', 1, 30, 7)
            st.markdown('</div>', unsafe_allow_html=True)

            # Split data
            split_date = '2014-12-31'
            train_data = df[:split_date]
            test_data = df[split_date:]

            # Run Model button
            if st.button('Run Models', key='run_forecast'):
                results = {}
                predictions = {}
                future_predictions = {}
                
                if use_xgboost:
                    with st.expander("XGBoost Progress", expanded=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Training
                        status_text.text("XGBoost: Fitting model...")
                        progress_bar.progress(0.3)
                        
                        # Run XGBoost model
                        xgb_model, xgb_metrics, xgb_predictions = run_xgboost_model(
                            df=df,
                            split_date=split_date,
                            target_column=target_col
                        )
                        progress_bar.progress(0.6)
                        
                        # Store results
                        predictions['XGBoost'] = xgb_predictions['MW_Prediction']
                        results['XGBoost'] = xgb_metrics
                        
                        # Future predictions
                        status_text.text("XGBoost: Making future predictions...")
                        future_dates = pd.date_range(
                            start=df.index[-1] + timedelta(hours=1),
                            periods=forecast_days*24,
                            freq='H'
                        )
                        future_df = pd.DataFrame(index=future_dates)
                        future_features = create_features(future_df)
                        future_predictions['XGBoost'] = pd.Series(
                            xgb_model.predict(future_features),
                            index=future_dates
                        )
                        progress_bar.progress(1.0)
                        status_text.text("XGBoost: Complete!")
                
                if use_prophet:
                    with st.expander("Prophet Progress", expanded=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Training
                        status_text.text("Prophet: Fitting model...")
                        prophet_model = ProphetModel()
                        prophet_model.fit(train_data, target_col)
                        progress_bar.progress(0.5)
                        
                        # Predictions
                        status_text.text("Prophet: Making predictions...")
                        forecast = prophet_model.predict(test_data.index)
                        predictions['Prophet'] = pd.Series(forecast['yhat'].values, index=test_data.index)
                        
                        # Future predictions
                        future_dates = pd.date_range(
                            start=df.index[-1] + timedelta(hours=1),
                            periods=forecast_days*24,
                            freq='H'
                        )
                        future_forecast = prophet_model.predict(future_dates)
                        future_predictions['Prophet'] = pd.Series(
                            future_forecast['yhat'].values,
                            index=future_dates
                        )
                        progress_bar.progress(0.8)
                        
                        # Metrics
                        status_text.text("Prophet: Calculating metrics...")
                        results['Prophet'] = prophet_model.evaluate(test_data, target_col)
                        progress_bar.progress(1.0)
                        status_text.text("Prophet: Complete!")

                # Plot results
                st.subheader('Forecast Results')
                fig = go.Figure()

                # Plot actual data
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=test_data[target_col],
                    mode='lines',
                    name='Actual'
                ))

                # Plot predictions and future predictions for each model
                colors = {'XGBoost': 'red', 'Prophet': 'green'}
                for model_name in predictions.keys():
                    # Plot predictions
                    fig.add_trace(go.Scatter(
                        x=predictions[model_name].index,
                        y=predictions[model_name],
                        mode='lines',
                        name=f'{model_name} Predicted',
                        line=dict(color=colors[model_name])
                    ))
                    
                    # Plot future predictions
                    fig.add_trace(go.Scatter(
                        x=future_predictions[model_name].index,
                        y=future_predictions[model_name],
                        mode='lines',
                        name=f'{model_name} Future Forecast',
                        line=dict(color=colors[model_name], dash='dash')
                    ))

                fig.update_layout(
                    title='Forecast Results',
                    xaxis_title='Date',
                    yaxis_title=target_col,
                    legend_title='Legend'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display metrics
                st.subheader('Model Performance')
                metrics_df = pd.DataFrame(results).T
                st.table(metrics_df.style.format("{:.4f}"))

                # Display raw data
                st.subheader('Raw Data')
                st.write(df)
        else:
            st.error("The file must contain a 'Datetime' column")

# Tab 2: Time Series Visualization
with tab2:
    st.title("Time Series Visualization")
    
    # File selection section
    st.markdown('<div class="stBlock">', unsafe_allow_html=True)
    st.subheader("Select Data Files")
    
    uploaded_files = st.file_uploader(
        "Upload multiple CSV files",
        type="csv",
        accept_multiple_files=True,
        key="viz_uploader"
    )
    
    if uploaded_files:
        st.write("### Selected Files Summary")
        file_summary = []
        dfs = {}
        selected_files = {}
        
        for file in uploaded_files:
            df = pd.read_csv(file)
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
            
            file_summary.append({
                'File Name': file.name,
                'Rows': df.shape[0],
                'Columns': df.shape[1],
                'Start Date': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None,
                'End Date': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
            })
            dfs[file.name] = df
        
        summary_df = pd.DataFrame(file_summary)
        st.dataframe(summary_df)
        
        st.write("### Select Files to Visualize")
        # Create checkboxes for file selection
        for name in dfs.keys():
            selected_files[name] = st.checkbox(f"Select {name}", key=f"select_{name}")
        
        # Add button to generate visualizations
        if st.button("Generate Visualizations", key="generate_viz"):
            # Only visualize selected files
            for name, selected in selected_files.items():
                if selected:
                    st.subheader(f"Time Series Plot for {name}")
                    try:
                        fig = plot_time_series(dfs[name])
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error plotting {name}: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Data Cleanup
with tab3:
    st.title("Data Cleanup and Analysis")
    
    cleanup_files = st.file_uploader(
        "Upload CSV files for cleanup",
        type="csv",
        accept_multiple_files=True,
        key="cleanup_uploader"
    )
    
    if cleanup_files:
        output_dir = st.text_input(
            "Enter output directory path (relative to project root)",
            "data/processed"
        )
        
        if st.button("Clean and Analyze Data", key="clean_data"):
            os.makedirs(output_dir, exist_ok=True)
            
            for file in cleanup_files:
                st.subheader(f"Processing {file.name}")
                
                # Read data
                df = pd.read_csv(file)
                
                # Show initial analysis
                st.write("### Initial Data Analysis")
                
                # DataFrame Info
                st.write("#### DataFrame Information")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                
                # Basic Statistics
                st.write("#### Basic Statistics")
                st.write(df.describe())
                
                # Missing Values Analysis
                st.write("#### Missing Values Analysis")
                missing_analysis = pd.DataFrame({
                    'Missing Count': df.isnull().sum(),
                    'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2),
                    'Zero Count': (df == 0).sum(),
                    'Zero Percentage': (df == 0).sum() / len(df) * 100
                })
                missing_analysis.index.name = 'Feature'
                
                # Format percentages
                missing_analysis['Missing Percentage'] = missing_analysis['Missing Percentage'].apply(lambda x: f"{x:.2f}%")
                missing_analysis['Zero Percentage'] = missing_analysis['Zero Percentage'].apply(lambda x: f"{x:.2f}%")
                
                # Only show features with missing values or zeros
                has_missing_or_zero = (missing_analysis['Missing Count'] > 0) | (missing_analysis['Zero Count'] > 0)
                if has_missing_or_zero.any():
                    st.write(missing_analysis[has_missing_or_zero])
                else:
                    st.write("No missing values or zeros found in the dataset.")
                
                # Clean data
                datetime_cols = [col for col in df.columns 
                               if 'date' in col.lower() or 'time' in col.lower()]
                if datetime_cols:
                    df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]])
                    df.set_index(datetime_cols[0], inplace=True)
                
                cleaned_df = clean_data(df)
                
                # Save cleaned data
                output_path = os.path.join(output_dir, f"cleaned_{file.name}")
                cleaned_df.to_csv(output_path)
                st.success(f"Cleaned data saved to {output_path}")
                
                # Show cleaned data summary
                st.write("### Cleaned Data Summary")
                st.write("First 5 rows:")
                st.write(cleaned_df.head())
                st.write("Last 5 rows:")
                st.write(cleaned_df.tail())
                
                # Time series visualization
                st.write("### Time Series Visualization")
                fig = plot_time_series(cleaned_df)
                st.pyplot(fig)
                plt.close()
                
                # Correlation heatmap
                st.write("### Feature Correlations")
                numeric_df = cleaned_df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("No numeric columns available for correlation heatmap")
