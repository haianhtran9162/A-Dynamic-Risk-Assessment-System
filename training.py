import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
model_path = config['output_model_path']

def train_model():
    """
        Function to train the model with the Logistic Regression model.
        Output: The trainedmodel.pkl file
    """
    logging.info("Load the data ingestion.")
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = df["exited"]
    
    # Use this logistic regression for training
    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=42, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    # Fit the logistic regression to your data
    logging.info("Training model")
    lr.fit(X, y)
    
    # Write the trained model to your workspace in a file called trainedmodel.pkl
    logging.info("Writing trained model to pkl file")
    pickle.dump(lr, open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb'))

if __name__ == '__main__':
    train_model()
 