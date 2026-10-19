import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def build_and_package_pipeline():
    print("Loading cleaned dataset...")
    data = pd.read_csv("cleaned_heart_disease.csv")
    
    X = data.drop(columns=["target"])
    y = data["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building full end-to-end Pipeline (Scaler + Random Forest)...")
    # Packaging the preprocessor and the classifier together ensures 100% reproducibility
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate quickly on test set
    test_score = pipeline.score(X_test, y_test)
    print(f"Pipeline Test Accuracy: {test_score:.4f}")

    # Create model output directory
    os.makedirs("models", exist_ok=True)
    model_path = "models/heart_disease_model.pkl"

    # Save the pipeline using pickle
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Task 4 Complete: Model successfully packaged and saved at '{model_path}'")

if __name__ == "__main__":
    build_and_package_pipeline()