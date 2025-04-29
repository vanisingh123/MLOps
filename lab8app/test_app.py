import requests

# Define the data to be passed to the model
data = {
    "feature_0": 5.1,
    "feature_1": 3.5,
    "feature_2": 1.4,
    "feature_3": 0.2
}

# Send a POST request to the FastAPI app
response = requests.post("http://127.0.0.1:8000/predict", json=data)

# Try to print the prediction or error
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Failed to decode JSON. Raw response:")
    print(response.text)
    print("Status code:", response.status_code)
