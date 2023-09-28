from datetime import datetime
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']

def score_model():
    """
        Function calulates the F1 score for the trained model on the data test.
        Output: Report of model score in latestscore.txt file.
    """
    logging.info('Load the data test.')
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = df_test[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_test = df_test["exited"]
    
    logging.info('Load the trained model.')
    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'),'rb'))
    
    logging.info("Predict the trained model with test data.")
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    logging.info("The F1 Score = {}".format(score))
    
    logging.info("Save the F1 Score to file.")
    log_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as file:
        file.write("Test time: {}\n".format(log_time))
        file.write("Model version: {}\n".format(str(os.path.join(output_model_path, 'trainedmodel.pkl'))))
        file.write("Data test: {}\n".format(os.path.join(test_data_path, 'testdata.csv')))
        file.write("The F1 Score = {}\n".format(score))
        file.write("________________________________________________________________")
    return score
    
if __name__ == '__main__':
    score_model()    
