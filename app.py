from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from diagnostics import model_predictions
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, outdated_packages_list

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

@app.route("/prediction", methods=['POST','OPTIONS'])
def post_predict():    
    try:    
        y_pred = model_predictions()
        y_pred = y_pred.tolist()
        return jsonify({
            "message": "SUCCESS",
            "code": 200,
            "data": y_pred
        })
    except Exception as e:
        return jsonify({
            "message": "ERROR",
            "code": 500,
            "data": str(e)
        })

@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():     
    try:   
        f1_score = score_model()
        return jsonify({
            "message": "SUCCESS",
            "code": 200,
            "data": {
                "f1_score": f1_score
            }
        })
    except Exception as e:
        return jsonify({
            "message": "ERROR",
            "code": 500,
            "data": str(e)
        })

@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_summary_statistics():   
    try:     
        summary_statistics, _ = dataframe_summary()
        return jsonify({
            "message": "SUCCESS",
            "code": 200,
            "data": {
                "summary_statistics": summary_statistics
            }
        })
    except Exception as e:
        return jsonify({
            "message": "ERROR",
            "code": 500,
            "data": str(e)
        })

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():  
    try:      
        _, missing_data = dataframe_summary()
        exec_time = execution_time()
        outdate_check = outdated_packages_list() 
        return jsonify({
            "message": "SUCCESS",
            "code": 200,
            "data": {
                "missing_data": missing_data,
                "execution_time": {
                    "ingestion_time": exec_time[0],
                    "training_time": exec_time[1]
                },
                "outdated_packages": outdate_check
            }
        })
    except Exception as e:
        return jsonify({
            "message": "ERROR",
            "code": 500,
            "data": str(e)
        })

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
