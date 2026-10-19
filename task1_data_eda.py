import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_data_pipeline():
    # 1. Data Acquisition
    print("Downloading Heart Disease UCI dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Defining the 14+ features based on the UCI documentation
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # The dataset uses '?' for missing values
    df = pd.read_csv(url, names=columns, na_values="?")

    # 2. Data Cleaning & Preprocessing
    print("Cleaning data and encoding features...")
    # Handle missing values by imputing the median for simplicity
    df.fillna(df.median(), inplace=True)

    # The assignment specifies a binary target (presence/absence of heart disease)
    # The raw UCI target ranges from 0 (absence) to 1, 2, 3, 4 (presence)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # 3. Exploratory Data Analysis (EDA)
    print("Generating professional visualizations...")
    os.makedirs("eda_plots", exist_ok=True)

    # Visualization A: Class Balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='target', palette='Set2')
    plt.title("Class Balance: Heart Disease (0 = Absence, 1 = Presence)")
    plt.savefig("eda_plots/class_balance.png", bbox_inches='tight')
    plt.close()

    # Visualization B: Histograms
    df.hist(figsize=(14, 12), bins=20, color='steelblue', edgecolor='black')
    plt.suptitle("Feature Histograms", fontsize=16)
    plt.savefig("eda_plots/histograms.png", bbox_inches='tight')
    plt.close()

    # Visualization C: Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.savefig("eda_plots/correlation_heatmap.png", bbox_inches='tight')
    plt.close()

    # Save the cleaned dataset for Phase 2
    df.to_csv("cleaned_heart_disease.csv", index=False)
    print("Task 1 Complete: Cleaned dataset saved as 'cleaned_heart_disease.csv'. Plots saved in 'eda_plots/'.")

if __name__ == "__main__":
    run_data_pipeline()