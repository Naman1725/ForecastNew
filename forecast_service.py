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
