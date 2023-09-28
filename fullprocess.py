

import json
import os
import pickle
import re
import logging
import pandas as pd

from datetime import datetime
from sklearn.metrics import f1_score
from ingestion import merge_multiple_dataframe

with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = config['prod_deployment_path']
input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']

def full_process():  
    ##################Check and read new data
    #first, read ingestedfiles.txt
    logging.info("Check had any update for data sources")
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        ingested_files = file.readlines()[1:2][0]
        ingested_files = ingested_files.split(':')[1]
        ingested_files = re.sub('[^a-zA-Z0-9 \/.,]', '', ingested_files)
        ingested_files = ingested_files.replace(' ', '')
        ingested_files_list = ingested_files.split(',')
        ingested_files_list
    file.close()
    
    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.tx
    data_source = []
    for file in os.listdir(input_folder_path):
        if str(file).endswith(".csv"):
            data_source.append(os.path.join(input_folder_path, file))
    
    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if set(data_source) == set(ingested_files_list):
        logging.info("Finish running flow with not updated data source")
        return None
    logging.info("Found new data sources. Start running flow")
    logging.info("Merge new dataset from new data sources")
    merge_multiple_dataframe()

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest
    logging.info("Checking for model drift")
    
    # Get the F1 score of old dataset
    f1_old = 0.0 
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
        ingested_files = file.readlines()[3:4][0]
        ingested_files = ingested_files.split("=")[1]
        f1_old = float(ingested_files.replace(" ", ""))
    file.close()
    logging.info(f"The F1 score deployment: {f1_old}" )

    # Get the F1 score of new dataset
    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    X_new = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y_new = df["exited"]
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'),'rb'))
    y_pred = model.predict(X_new)
    f1_new = f1_score(y_new, y_pred)
    logging.info(f"The new F1 score: {f1_new}")
    
    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if f1_new >= f1_old:
        logging.info("The model is still working well without mode drift")
        return None
    
    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    # Re-train model
    logging.info("Start re-training the model")
    os.system('python3 training.py')
    
    # Re-calculate scoring model
    logging.info("Start re-calculate scoring new model")
    os.system('python3 scoring.py')
    
    # Re-deployment
    logging.info("Start re-deploy new model")
    os.system('python3 deployment.py')

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    logging.info("Start re-diagnostic and re-run reporting")
    os.system('python3 diagnostics.py')
    os.system('python3 reporting.py')
    logging.info("Finish update model, the new model had been released")
    

if __name__ == '__main__':
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s ',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename= 'log_run_full_process.log',
                        filemode='a')
    logging.info("____________________________________")
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Running new process: {date_time}")
    full_process()





