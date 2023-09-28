import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = config['test_data_path']
output_model_path = config['output_model_path']

def score_model():
    """
        Function calculate a confusion matrix using the test data and the deployed model
        Output: the confusion matrix (confusionmatrix.png)
    """
    logging.info("Load trained model")
    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'),'rb'))
    
    logging.info("Load data test")
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = df_test[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = df_test["exited"]
    
    logging.info("Predict the data test")
    y_pred = model.predict(X_test)
    
    logging.info("Create confusion metric")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12,8))
    ax = sns.heatmap(cm, annot=True, fmt='d', )
    ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['Negative', 'Positive'])
    ax.set_ylabel("Actual", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['Negative', 'Positive'])
    ax.set_title("Confusion Matrix", fontsize=14, pad=20)
    plt.savefig(os.path.join(output_model_path, 'confusion_matrix.png'))
    
if __name__ == '__main__':
    score_model()
