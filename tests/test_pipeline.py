import os
import pandas as pd
import pickle

def test_data_exists():
    """Test if the cleaned data file was created successfully."""
    assert os.path.exists("cleaned_heart_disease.csv"), "Cleaned data file is missing!"
    
    # Check if data has the correct columns
    df = pd.read_csv("cleaned_heart_disease.csv")
    assert 'target' in df.columns, "Target column is missing from the dataset!"
    assert not df.empty, "The dataset is empty!"

def test_model_prediction():
    """Test if the model loads and can make a prediction on dummy data."""
    assert os.path.exists("models/heart_disease_model.pkl"), "Model file is missing!"
    
    with open("models/heart_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    # Get a single row of real data to test the model
    df = pd.read_csv("cleaned_heart_disease.csv")
    X_sample = df.drop(columns=["target"]).iloc[[0]]
    
    # Model should output either 0 or 1
    prediction = model.predict(X_sample)
    assert prediction[0] in [0, 1], "Prediction must be 0 or 1!"