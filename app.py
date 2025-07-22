# app.py
from flask import Flask, request, jsonify
from forecast_service import run_forecast_pipeline
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "KPI Forecasting API is running. Use /forecast endpoint."

@app.route('/forecast', methods=['POST'])
def forecast():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        buffer = file.read()
        country = request.form.get('country')
        tech = request.form.get('tech')
        zone = request.form.get('zone')
        kpi = request.form.get('kpi')
        months = int(request.form.get('months', 3))

        if not all([country, tech, zone, kpi]):
            return jsonify({'error': 'Missing one or more required fields: country, tech, zone, kpi'}), 400

        plot_json, summary, err = run_forecast_pipeline(buffer, country, tech, zone, kpi, months)

        if err:
            return jsonify({'error': err}), 400

        return jsonify({
            'summary': summary,
            'plot': plot_json
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
