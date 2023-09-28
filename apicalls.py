import json
import requests
import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = config['output_model_path']

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
logging.info("POST request to '/prediction'")
prediction = requests.post(f'{URL}/prediction').text

logging.info("GET request to '/scoring'")
scoring = requests.get(f'{URL}/scoring').text

logging.info("GET request to '/summarystats'")
summarystats = requests.get(f'{URL}/summarystats').text

logging.info("GET request to '/diagnostics'")
diagnostics = requests.get(f'{URL}/diagnostics').text

# Combine all API responses
logging.info("Log response to apireturns.txt file")
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as file:
    file.write("Calling test API endpoints\n")
    file.write("________________________________________________________________")
    file.write("\nResponse when call API '/prediction' method POST\n")
    file.write(prediction)
    file.write("________________________________________________________________")
    file.write("\nResponse when call API '/scoring' method GET\n")
    file.write(scoring)
    file.write("________________________________________________________________")
    file.write("\nResponse when call API '/summarystats' method GET\n")
    file.write(summarystats)
    file.write("________________________________________________________________")
    file.write("\nResponse when call API '/diagnostics' method GET\n")
    file.write(diagnostics)
    file.write("________________________________________________________________")
file.close()



