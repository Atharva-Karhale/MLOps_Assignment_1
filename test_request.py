import requests

# Dummy patient health data
patient_data = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}

print("Sending patient data to Docker API...")
response = requests.post("http://127.0.0.1:5000/predict", json=patient_data)

print("API Response:")
print(response.json())
