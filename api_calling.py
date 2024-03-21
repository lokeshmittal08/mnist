import requests

# Specify the URL of the API endpoint
url = "http://localhost:8000/run_train/"

# Define the file to upload (configuration file)
files = {'config_file': open('new_config.yaml', 'rb')}  # Replace 'config.yaml' with the path to your configuration file

# Make the POST request
response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    print("Model training completed successfully.")
else:
    print("Error:", response.text)
