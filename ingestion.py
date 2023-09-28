import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

def merge_multiple_dataframe():
    """
        The function will read all the dataset and merge to the final datasets
        Output: final dataset and metadata ingestion information
    """
    try:
        logging.info("Start merge all datasets.")
        df = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees', 'exited'])
        list_sub_df = []
        for file in os.listdir(input_folder_path):
            # Get only the file end with .csv
            if str(file).endswith(".csv"):
                # Get the file path
                file_path = os.path.join(input_folder_path, file)
                list_sub_df.append(file_path)
                df_sub = pd.read_csv(file_path)
                df = df.append(df_sub)
        logging.info("Merge all datasets finished")
        
        logging.info("Remove duplicate on merge dataset")
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=1, inplace=True)
        
        logging.info("Saving the data ingestion")
        df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
        
        logging.info("Saving the metadata ingestion")
        log_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
            file.write("Ingestion time: {}\n".format(log_time))
            file.write("List input file: {}\n".format(str(list_sub_df)))
            file.write("Output file: {}\n".format(os.path.join(output_folder_path, 'finaldata.csv')))
            file.write("Output file shape: {}\n".format(df.shape))
            file.write("________________________________________________________________")
    except Exception as e:
        logging.error("Error writing final data: ", e)
        
if __name__ == '__main__':
    merge_multiple_dataframe()
