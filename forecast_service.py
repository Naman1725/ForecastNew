"""
# forecast_service.py
import zipfile
import io
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import json

def extract_data_from_zip(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        all_dfs = []
        for file in archive.namelist():
            if file.endswith(".xlsx"):
                try:
                    # Attempt to parse date from filename (e.g., Jan2023.xlsx)
                    try:
                        date = pd.to_datetime(file.split(".")[0], format="%b%Y")
                    except Exception:
                        # Fallback if filename is not in expected format
                        date = pd.Timestamp.now()

                    df = pd.read_excel(archive.open(file), engine='openpyxl')
                    df['Month'] = date
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Skipping file {file} due to error: {e}")
                    continue
        if not all_dfs:
            raise ValueError("No valid Excel files found.")
        return pd.concat(all_dfs, ignore_index=True)

def filter_and_prepare(df, country, tech, zone, kpi):
    df = df[
        (df['Country'] == country) &
        (df['Technology'] == tech) &
        (df['Zone'] == zone) &
        (df['KPI'] == kpi)
    ]
    if df.empty:
        raise ValueError("No matching KPI records found.")

    df = df[['Month', 'Actual Value MAPS Networks']]
    df = df.groupby('Month').mean().reset_index()
    df.rename(columns={'Month': 'ds', 'Actual Value MAPS Networks': 'y'}, inplace=True)
    return df

def forecast_with_lr(df, months=3):
    df['t'] = np.arange(len(df))
    X = df[['t']]
    y = df['y']
    model = LinearRegression()
    model.fit(X, y)
    future_t = np.arange(len(df), len(df) + months).reshape(-1, 1)
    future_dates = pd.date_range(start=df['ds'].max() + pd.offsets.MonthBegin(), periods=months, freq='MS')
    y_pred = model.predict(future_t)
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': y_pred,
        'yhat_upper': y_pred + 0.2,
        'yhat_lower': y_pred - 0.2
    })
    return forecast_df

def generate_plot(df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title='KPI Forecast', xaxis_title='Date', yaxis_title='Value')
    return fig.to_json()

def run_forecast_pipeline(zip_bytes, country, tech, zone, kpi, months=3):
    try:
        df = extract_data_from_zip(zip_bytes)
        ts_df = filter_and_prepare(df, country, tech, zone, kpi)
        forecast_df = forecast_with_lr(ts_df, months)
        plot_json = generate_plot(ts_df, forecast_df)
        summary = forecast_df.to_dict(orient='records')
        return plot_json, summary, None
    except Exception as e:
        return None, None, str(e)
"""



"""
# forecast_service.py

import zipfile
import io
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import json

def extract_data_from_zip(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        all_dfs = []
        for n, file in enumerate(archive.namelist()):
            if file.endswith(".xlsx"):
                try:
                    # Try to parse date from filename (e.g., Jul2023.xlsx)
                    date = pd.to_datetime(file.split(".")[0], format="%b%Y")
                except:
                    # Fallback: generate sequential months starting from a base
                    base_date = pd.Timestamp("2023-01-01")
                    date = base_date + pd.offsets.MonthBegin(n)

                try:
                    df = pd.read_excel(archive.open(file))
                    df['Month'] = date
                    all_dfs.append(df)
                except Exception:
                    continue

        if not all_dfs:
            raise ValueError("No valid Excel files found.")
        return pd.concat(all_dfs, ignore_index=True)

def filter_and_prepare(df, country, tech, zone, kpi):
    df = df[
        (df['Country'] == country) &
        (df['Technology'] == tech) &
        (df['Zone'] == zone) &
        (df['KPI'] == kpi)
    ]
    if df.empty:
        raise ValueError("No matching KPI records found.")

    df = df[['Month', 'Actual Value MAPS Networks']]
    df = df.groupby('Month').mean().reset_index()
    df.rename(columns={'Month': 'ds', 'Actual Value MAPS Networks': 'y'}, inplace=True)
    return df

def forecast_with_lr(df, months=3):
    df['t'] = np.arange(len(df))
    X = df[['t']]
    y = df['y']
    model = LinearRegression()
    model.fit(X, y)

    # Forecast
    future_t = np.arange(len(df), len(df) + months).reshape(-1, 1)
    future_dates = pd.date_range(start=df['ds'].max() + pd.offsets.MonthBegin(), periods=months, freq='MS')
    y_pred = model.predict(future_t)

    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': y_pred,
        'yhat_upper': y_pred + 0.2,
        'yhat_lower': y_pred - 0.2
    })
    return forecast_df

def generate_plot(df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
    fig.update_layout(title='KPI Forecast', xaxis_title='Date', yaxis_title='Value')
    return fig.to_json()

def run_forecast_pipeline(zip_bytes, country, tech, zone, kpi, months=3):
    try:
        df = extract_data_from_zip(zip_bytes)
        ts_df = filter_and_prepare(df, country, tech, zone, kpi)
        forecast_df = forecast_with_lr(ts_df, months)
        plot_json = generate_plot(ts_df, forecast_df)
        summary = forecast_df.to_dict(orient='records')
        return plot_json, summary, None
    except Exception as e:
        return None, None, str(e)

"""



import zipfile
import io
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime

def extract_data_from_zip(zip_bytes):
    """Robust extraction with proper date handling and error logging"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        all_dfs = []
        valid_files = sorted([f for f in archive.namelist() if f.endswith(".xlsx")])
        
        if not valid_files:
            raise ValueError("No Excel files found in ZIP archive")
        
        for idx, file in enumerate(valid_files):
            try:
                # Extract date from filename (format: MonthYYYY.xlsx)
                month_str, year_str = file.split('.')[0][:3], file.split('.')[0][3:]
                date = datetime.strptime(f"{month_str} {year_str}", "%b %Y")
                
                with archive.open(file) as excel_file:
                    df = pd.read_excel(excel_file)
                    df['Month'] = date
                    all_dfs.append(df)
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

        return pd.concat(all_dfs, ignore_index=True)

def filter_and_prepare(df, country, tech, zone, kpi):
    """Improved filtering and time series preparation"""
    # Validate input parameters
    valid_params = {
        'Country': country,
        'Technology': tech,
        'Zone': zone,
        'KPI': kpi
    }
    
    # Filter and validate
    filtered = df.copy()
    for col, value in valid_params.items():
        if value:
            filtered = filtered[filtered[col] == value]
    
    if filtered.empty:
        raise ValueError("No data matching the specified filters")
    
    # Prepare time series
    ts = (
        filtered.groupby('Month')
        ['Actual Value MAPS Networks']
        .mean()
        .reset_index()
        .rename(columns={'Month': 'ds', 'Actual Value MAPS Networks': 'y'})
        .sort_values('ds')
    )
    
    return ts

def forecast_with_lr(df, months=3):
    """Enhanced forecasting with data validation"""
    if len(df) < 2:
        raise ValueError("At least 2 data points required for forecasting")
    
    # Create time index
    df = df.copy()
    df['t'] = np.arange(len(df))
    
    # Train model
    model = LinearRegression()
    model.fit(df[['t']], df['y'])
    
    # Generate forecast dates
    last_date = df['ds'].max()
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=months,
        freq='MS'
    )
    
    # Predict values
    future_t = np.arange(len(df), len(df) + months)
    y_pred = model.predict(future_t.reshape(-1, 1))
    
    # Calculate confidence interval (std of residuals)
    residuals = df['y'] - model.predict(df[['t']])
    std_dev = residuals.std()
    
    return pd.DataFrame({
        'ds': future_dates,
        'yhat': y_pred,
        'yhat_upper': y_pred + 1.96 * std_dev,
        'yhat_lower': y_pred - 1.96 * std_dev
    })

def generate_plot(actual_df, forecast_df):
    """Create interactive plot with proper styling"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=actual_df['ds'],
        y=actual_df['y'],
        name='Actual',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Forecasted values
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        name='Forecast',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence'
    ))
    
    fig.update_layout(
        title='KPI Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig.to_json()

def run_forecast_pipeline(zip_bytes, country, tech, zone, kpi, months=3):
    """Main pipeline with comprehensive error handling"""
    try:
        # Step 1: Extract and process data
        df = extract_data_from_zip(zip_bytes)
        
        # Step 2: Filter and prepare time series
        ts_df = filter_and_prepare(df, country, tech, zone, kpi)
        
        # Step 3: Generate forecast
        forecast_df = forecast_with_lr(ts_df, months)
        
        # Step 4: Create visualization
        plot_json = generate_plot(ts_df, forecast_df)
        
        # Step 5: Prepare summary
        forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
        summary = forecast_df.to_dict(orient='records')
        
        return plot_json, summary, None
        
    except Exception as e:
        error_msg = f"Forecast failed: {str(e)}"
        print(error_msg)
        return None, None, error_msg

