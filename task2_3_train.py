import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

import mlflow
import mlflow.sklearn

def train_and_track():
    # 1. Load the cleaned data from Task 1
    if not os.path.exists("cleaned_heart_disease.csv"):
        print("Error: cleaned_heart_disease.csv not found! Run task1_data_eda.py first.")
        return

    data = pd.read_csv("cleaned_heart_disease.csv")
    
    # Separate input features (X) and target outcome (y)
    X = data.drop(columns=["target"])
    y = data["target"]

    # 2. Split into 80% Training and 20% Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Feature Scaling (normalizing numerical values)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up MLflow tracking
    mlflow.set_experiment("Heart_Disease_Prediction")

    # Define the two candidate models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    # Make a temporary directory for evaluation plots
    os.makedirs("metrics_plots", exist_ok=True)

    # 4. Train, evaluate, and track both models
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"\n--- Training {model_name} ---")

            # Train on scaled data
            model.fit(X_train_scaled, y_train)

            # Predict labels and probability scores
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate required metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"ROC-AUC:   {auc:.4f}")

            # Log metrics to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", auc)

            # Generate and log ROC Curve plot
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            
            plot_path = f"metrics_plots/roc_{model_name}.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

            mlflow.log_artifact(plot_path)

            # Log the trained model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"Logged {model_name} artifacts to MLflow.")

    print("\nTasks 2 & 3 Complete!")

if __name__ == "__main__":
    train_and_track()