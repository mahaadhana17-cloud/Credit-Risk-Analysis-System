import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
csv_files = list(Path(".").glob("*.csv"))

if not csv_files:
    print("Dataset not found. Please upload the CSV file.")
else:
    df = pd.read_csv(csv_files[0])

    # Display dataset
    print("Dataset loaded successfully!")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    # Correlation analysis
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Credit Risk Correlation Analysis")
        plt.tight_layout()
        plt.show()

    print("\nExploratory Data Analysis Completed!"
