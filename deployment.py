import shutil
import os
from sklearn.model_selection import train_test_split
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

def store_model_into_pickle():
    """
        Function copy the latest model, log score file, log ingest metadata to the deployment directory
    """
    try:
        logging.info("Move model and result log files to the deployment directory")
        
        logging.info("Start copying metadate ingestion.")
        shutil.copy(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), prod_deployment_path)
        
        logging.info("Start copying model pkl.")
        shutil.copy(os.path.join(output_model_path, 'trainedmodel.pkl'), prod_deployment_path)
        
        logging.info("Start copying model scoring.")
        shutil.copy(os.path.join(output_model_path, 'latestscore.txt'), prod_deployment_path)
        
    except Exception as e:
        logging.error("Error when move model and result log files to the deployment directory : ", e)
        
if __name__ == '__main__':
    store_model_into_pickle()

