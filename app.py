from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import pandas as pd
import json
import os
from flask_cors import CORS

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Enable CORS for React frontend
CORS(app)

# API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data_json = request.get_json()
        
        data = CustomData(
            gender=data_json.get('gender'),
            race_ethnicity=data_json.get('ethnicity'),
            parental_level_of_education=data_json.get('parental_level_of_education'),
            lunch=data_json.get('lunch'),
            test_preparation_course=data_json.get('test_preparation_course'),
            reading_score=float(data_json.get('reading_score')),
            writing_score=float(data_json.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        
        # Get prediction from best model
        results = predict_pipeline.predict(pred_df)
        
        # Get predictions from all models
        all_predictions = predict_pipeline.predict_all_models(pred_df)
        
        # Load model report for R² scores
        model_report = None
        report_path = os.path.join('artifacts', 'model_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                model_report = json.load(f)
        
        # Combine predictions with accuracy data
        if all_predictions and model_report:
            for pred in all_predictions:
                for model in model_report['models']:
                    if model['name'] == pred['name']:
                        pred['r2_score'] = model['r2_score']
                        pred['accuracy_percentage'] = model['accuracy_percentage']
                        break
            
            # Sort by R² score (highest first)
            all_predictions.sort(key=lambda x: x.get('r2_score', 0), reverse=True)
        
        return jsonify({
            'success': True,
            'prediction': float(results[0]),
            'all_predictions': all_predictions,
            'best_model': model_report.get('best_model') if model_report else None
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for React app
@app.route('/react')
def react_app():
    return send_from_directory('frontend', 'index.html')

# Route for prediction form and result
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:\n", pred_df)

        predict_pipeline = PredictPipeline()
        
        # Get prediction from best model
        results = predict_pipeline.predict(pred_df)
        
        # Get predictions from all models
        all_predictions = predict_pipeline.predict_all_models(pred_df)
        
        # Load model report for R² scores
        model_report = None
        report_path = os.path.join('artifacts', 'model_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                model_report = json.load(f)
        
        # Combine predictions with accuracy data
        if all_predictions and model_report:
            for pred in all_predictions:
                for model in model_report['models']:
                    if model['name'] == pred['name']:
                        pred['r2_score'] = model['r2_score']
                        pred['accuracy_percentage'] = model['accuracy_percentage']
                        break
            
            # Sort by R² score (highest first)
            all_predictions.sort(key=lambda x: x.get('r2_score', 0), reverse=True)

        return render_template('home.html', 
                             results=results[0], 
                             model_report=model_report,
                             all_predictions=all_predictions)

# Run the app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)