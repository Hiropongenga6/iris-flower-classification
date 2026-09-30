"""
Data Science Project: Iris Flower Classification

This is a complete machine learning pipeline demonstrating:
- Data exploration and preprocessing
- Train/test split and cross-validation
- Multiple model training (Logistic Regression, Random Forest, SVM)
- Model evaluation (accuracy, precision, recall, F1-score, ROC-AUC)
- Hyperparameter tuning
- Feature importance analysis
- Results and predictions saved to files

Perfect for portfolios and job applications requiring ML skills.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from itertools import combinations


def load_iris_data() -> pd.DataFrame:
    """Load the iris dataset and return it as a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def show_basic_information(df: pd.DataFrame) -> None:
    """Print basic information that beginners should inspect first."""
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(list(df.columns))

    print("\nSummary statistics:")
    print(df.describe())


def show_grouped_summary(df: pd.DataFrame) -> None:
    """Show average values grouped by species."""
    print("\nAverage feature values by species:")
    grouped = df.groupby("species").mean()
    print(grouped)


def save_dataset_csv(df: pd.DataFrame, path: Path) -> None:
    """Save the dataset to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved dataset CSV to: {path}")


def plot_histograms(df: pd.DataFrame, save_dir: Path) -> None:
    """Create histograms for each numeric feature."""
    save_dir.mkdir(parents=True, exist_ok=True)
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        plt.figure(figsize=(6, 4))
        df[column].hist(bins=15, color="#4c72b0", edgecolor="black")
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        filepath = save_dir / f"histogram_{column.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"Saved histogram: {filepath}")

