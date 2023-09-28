
import pickle
import subprocess
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
output_model_path = config['output_model_path']
output_folder_path = config['output_folder_path']

def model_predictions():
    """
        Function read production model, data test and calulate predictions
        Return: the y predictions (list of predictions)
    """
    logging.info("Reading data test")
    df_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = df_test[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    
    logging.info("Load the production model")
    model = pickle.load(open(os.path.join(output_model_path, 'trainedmodel.pkl'),'rb'))
    
    logging.info("Predict the data test")
    y_pred = model.predict(X_test)
    return y_pred

def dataframe_summary():
    """
        Fuction calculator summary statistics of the dataframe
        Return: 
            list value of mean, median, standard deviation (list)
            dict value of Nan value in dataset (dict)
    """
    logging.info("Load the ingestion data")
    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    logging.info("Calculating the mean, median and standard deviation scores")
    mean = X.mean()
    median = X.median()
    std = X.std()
    summary_statistics = []
    for col in X.columns:
        summary_statistics.append({
            col : {
                "mean_value": mean[col],
                "median_value": median[col],
                "std": std[col]
            }
        })
    logging.info("Calculating the missing data value percentage")
    total_missing = df.isna().sum()
    precentage_missing = total_missing/df.shape[0]
    return summary_statistics, precentage_missing.to_dict()

def execution_time():
    """
        Function calculate timing of training.py and ingestion.py
        Return:
            a list of 2 timing values in seconds (list)
    """
    logging.info("Calculate timing of ingestion.py")
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    
    logging.info("Calculate timing of traning.py")
    start_time = timeit.default_timer()
    os.system('python3 training.py')
    training_time = timeit.default_timer() - start_time
    
    return [ingestion_time, training_time]
    
def outdated_packages_list():
    """
        Get the list of outdated packages from the pip installation 
    """
    logging.info("Get the list of outdated packages from the pip installation")
    outdate_list = subprocess.run(['pip', 'list', '--outdated'], stdout=subprocess.PIPE, text=True)
    outdated_packages = []
    for line in outdate_list.stdout.split('\n')[2:-1]:
        outdated_packages.append({
            "package" : line.split()[0],
            "current_version" : line.split()[1],
            'latest_version' : line.split()[2]
        })
    return outdated_packages

if __name__ == '__main__':
    print("Make predictions for an input dataset using the current deployed model:")
    print(model_predictions())
    print("________________________________________________________________")
    
    summary_statistics, precentage_missing = dataframe_summary()
    print("Calculate summary statistics (mean, median, and standard deviation) for each numeric column in a dataset:")
    print(summary_statistics)
    print("________________________________________________________________")
    print("Calculate the percent of each column that consists of NA values:")
    print(precentage_missing)
    print("________________________________________________________________")
    
    count_time = execution_time()
    print("Measure the timing of important ML tasks (data ingestion and training):")
    print("Ingestion time: ", count_time[0])
    print("Training time: ", count_time[1])
    print("________________________________________________________________")
    
    print("Check whether the module dependencies are up-to-date:")
    print(outdated_packages_list())
    print("________________________________________________________________")




    
