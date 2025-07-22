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



# forecast_service.py
import zipfile
import io
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go
import json
import os
import re
from datetime import datetime

def extract_data_from_zip(zip_bytes):
    """Robust extraction with flexible date handling and file discovery"""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        all_dfs = []
        # Find all Excel files in any directory structure
        excel_files = [f for f in archive.namelist() if f.lower().endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            raise ValueError("No Excel files found in ZIP archive")
        
        print(f"Found {len(excel_files)} Excel files in ZIP")
        
        for idx, file in enumerate(excel_files):
            try:
                # Extract base filename without extension
                filename = os.path.basename(file)
                filename_no_ext = os.path.splitext(filename)[0]
                
                # Try to extract date from filename using regex
                date_match = re.search(
                    r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s?\d{4}\b|\d{4}-\d{2}-\d{2}|\d{8})', 
                    filename_no_ext, 
                    re.IGNORECASE
                )
                
                if date_match:
                    try:
                        date_str = date_match.group(0)
                        # Clean date string (remove spaces, dashes, etc.)
                        clean_date = re.sub(r'[^\w]', '', date_str)
                        date = pd.to_datetime(clean_date, errors='coerce', format='%b%Y')
                        if pd.isnull(date):
                            date = pd.to_datetime(clean_date, errors='coerce')
                    except:
                        date = None
                else:
                    date = None
                
                # Fallback to modification date or index-based date
                if not date or pd.isnull(date):
                    file_info = archive.getinfo(file)
                    mod_date = datetime(*file_info.date_time)
                    date = mod_date if mod_date.year > 2000 else pd.Timestamp("2023-01-01") + pd.DateOffset(months=idx)
                
                print(f"Processing {file} with date {date}")
                
                with archive.open(file) as excel_file:
                    # Try reading with different engines
                    try:
                        df = pd.read_excel(excel_file, engine='openpyxl')
                    except Exception as e1:
                        print(f"Openpyxl failed, trying xlrd: {str(e1)}")
                        excel_file.seek(0)  # Reset file pointer
                        try:
                            df = pd.read_excel(excel_file, engine='xlrd')
                        except Exception as e2:
                            print(f"xlrd failed: {str(e2)}")
                            raise ValueError(f"Could not read {file} with any engine")
                    
                    # Add month column and append to list
                    df['Month'] = date
                    all_dfs.append(df)
                    
            except Exception as e:
                print(f"Skipping file {file} due to error: {str(e)}")
                continue

        if not all_dfs:
            raise ValueError("All Excel files failed to process")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Successfully processed {len(all_dfs)} files, total rows: {len(combined_df)}")
        return combined_df

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
        if value and col in filtered.columns:
            filtered = filtered[filtered[col] == value]
        else:
            raise ValueError(f"Column '{col}' not found in dataset")
    
    if filtered.empty:
        raise ValueError(f"No data matching: Country={country}, Technology={tech}, Zone={zone}, KPI={kpi}")
    
    # Prepare time series
    if 'Actual Value MAPS Networks' not in filtered.columns:
        raise ValueError("'Actual Value MAPS Networks' column not found")
    
    ts = (
        filtered.groupby('Month')
        ['Actual Value MAPS Networks']
        .mean()
        .reset_index()
        .rename(columns={'Month': 'ds', 'Actual Value MAPS Networks': 'y'})
        .sort_values('ds')
    )
    
    if ts.empty:
        raise ValueError("Time series data is empty after grouping")
    
    return ts

def forecast_with_lr(df, months=3):
    """Enhanced forecasting with data validation"""
    if len(df) < 2:
        raise ValueError(f"At least 2 data points required for forecasting (only {len(df)} found)")
    
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
    
    # Handle case with no variance
    if std_dev == 0:
        std_dev = 0.1 * abs(y_pred[0]) if y_pred[0] != 0 else 0.1
    
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
        print("Starting ZIP extraction")
        df = extract_data_from_zip(zip_bytes)
        print(f"Extracted data with columns: {df.columns.tolist()}")
        
        # Step 2: Filter and prepare time series
        print(f"Filtering for: {country}, {tech}, {zone}, {kpi}")
        ts_df = filter_and_prepare(df, country, tech, zone, kpi)
        print(f"Time series prepared with {len(ts_df)} points")
        
        # Step 3: Generate forecast
        print(f"Forecasting {months} months")
        forecast_df = forecast_with_lr(ts_df, months)
        print("Forecast completed")
        
        # Step 4: Create visualization
        plot_json = generate_plot(ts_df, forecast_df)
        print("Plot generated")
        
        # Step 5: Prepare summary
        forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
        summary = forecast_df.to_dict(orient='records')
        
        return plot_json, summary, None
        
    except Exception as e:
        error_msg = f"Forecast failed: {str(e)}"
        print(error_msg)
        return None, None, error_msg

