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


def plot_scatter(df: pd.DataFrame, save_dir: Path) -> None:
    """Create scatter plots comparing all possible feature combinations."""
    save_dir.mkdir(parents=True, exist_ok=True)

    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    colors = {
        "setosa": "#1f77b4",
        "versicolor": "#ff7f0e",
        "virginica": "#2ca02c"
    }

    markers= {"setosa": "o", "versicolor": "s", "virginica": "^"}


    # Generate all possible pairs of features
    for x_feature, y_feature in combinations(features, 2):

        plt.figure(figsize=(7, 6))

        for species_name, group in df.groupby("species"):
            plt.scatter(
                group[x_feature],
                group[y_feature],
                label=species_name,
                color=colors.get(species_name, "gray"),
                alpha=0.8,
                marker=markers.get(species_name, "o"),
                edgecolors="w",
                s=80,
            )

        plt.title(f"{x_feature} vs {y_feature}")
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)

        # Create a safe filename
        filename = (
            f"{x_feature}_vs_{y_feature}"
            .replace(" ", "_")
            .replace("(cm)", "")
        )

        filepath = save_dir / f"{filename}.png"

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        print(f"Saved scatter plot: {filepath}")

def main() -> None:
    project_root = Path(__file__).parent
    output_dir = project_root / "plots"
    csv_path = project_root / "iris_dataset.csv"
    results_path = project_root / "model_results.json"

    # ========== STEP 1: DATA LOADING ==========
    print("=" * 60)
    print("STEP 1: Loading and Exploring Data")
    print("=" * 60)
    df = load_iris_data()
    show_basic_information(df)
    show_grouped_summary(df)
    save_dataset_csv(df, csv_path)

    # ========== STEP 2: DATA VISUALIZATION ==========
    print("\n" + "=" * 60)
    print("STEP 2: Creating Exploratory Visualizations")
    print("=" * 60)
    plot_histograms(df, output_dir)
    plot_scatter(df, output_dir)

    # ========== STEP 3: DATA PREPROCESSING ==========
    print("\n" + "=" * 60)
    print("STEP 3: Data Preprocessing and Feature Scaling")
    print("=" * 60)
    X = df.drop("species", axis=1).values
    y = df["species"].values
    
    # Standardize features (important for SVM and Logistic Regression)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Features scaled using StandardScaler")
    print(f"Feature scaling mean (should be ~0): {X_train_scaled.mean(axis=0)}")
    print(f"Feature scaling std (should be ~1): {X_train_scaled.std(axis=0)}")

    # ========== STEP 4: TRAIN/TEST SPLIT ==========
    print("\n" + "=" * 60)
    print("STEP 4: Splitting Data into Train and Test Sets")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {np.bincount(pd.Categorical(y_train, categories=np.unique(y_train)).codes)}")

    # ========== STEP 5: MODEL TRAINING ==========
    print("\n" + "=" * 60)
    print("STEP 5: Training Multiple Models")
    print("=" * 60)
    
    # Model 1: Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=200, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

    # Model 2: Random Forest
    print("\n--- Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Model 3: Support Vector Machine
    print("\n--- Support Vector Machine (SVM) ---")
    svm_model = SVC(kernel="rbf", random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy: {svm_acc:.4f}")

    # Model 4: K-Nearest Neighbors
    print("--- K-Nearest Neighbors (KNN) ---")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Accuracy: {knn_acc:.4f}")

    # ========== STEP 6: CROSS-VALIDATION ==========
    print("\n" + "=" * 60)
    print("STEP 6: Cross-Validation (5-Fold)")
    print("=" * 60)
    
    cv_scores_lr = cross_val_score(LogisticRegression(max_iter=200), X_train_scaled, y_train, cv=5)
    cv_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=100), X_train_scaled, y_train, cv=5)
    cv_scores_svm = cross_val_score(SVC(kernel="rbf"), X_train_scaled, y_train, cv=5)
    cv_scores_knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train, cv=5)
    
    print(f"Logistic Regression CV scores: {cv_scores_lr} | Mean: {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")
    print(f"Random Forest CV scores: {cv_scores_rf} | Mean: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
    print(f"SVM CV scores: {cv_scores_svm} | Mean: {cv_scores_svm.mean():.4f} ± {cv_scores_svm.std():.4f}")
    print(f"KNN CV scores: {cv_scores_knn} | Mean: {cv_scores_knn.mean():.4f} ± {cv_scores_knn.std():.4f}")

    # ========== STEP 7: HYPERPARAMETER TUNING ==========
    print("\n" + "=" * 60)
    print("STEP 7: Hyperparameter Tuning (Random Forest)")
    print("=" * 60)
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred_best)
    print(f"Test accuracy with best model: {best_acc:.4f}")

    # ========== STEP 8: DETAILED EVALUATION ==========
    print("\n" + "=" * 60)
    print("STEP 8: Detailed Model Evaluation")
    print("=" * 60)
    
    # Use the best model (tuned Random Forest)
    print("\nBest Model (Tuned Random Forest) - Classification Report:")
    print(classification_report(y_test, y_pred_best))
    
    print("\nConfusion Matrix:")
    labels=["setosa", "versicolor", "virginica"]
    cm = confusion_matrix(y_test, y_pred_best, labels=labels)
    
    print(cm)
    
    precision = precision_score(y_test, y_pred_best, average="weighted")
    recall = recall_score(y_test, y_pred_best, average="weighted")
    f1 = f1_score(y_test, y_pred_best, average="weighted")
    
    print(f"\nWeighted Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # ========== STEP 9: FEATURE IMPORTANCE ==========
    print("\n" + "=" * 60)
    print("STEP 9: Feature Importance Analysis")
    print("=" * 60)
    
    feature_names = df.drop("species", axis=1).columns
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values("Importance", ascending=False)
    
    print(importance_df.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="#4c72b0")
    plt.xlabel("Importance")
    plt.title("Feature Importance (Tuned Random Forest)")
    plt.tight_layout()
    importance_plot_path = output_dir / "feature_importance.png"
    plt.savefig(importance_plot_path)
    plt.close()
    print(f"\nFeature importance plot saved: {importance_plot_path}")

    # ========== STEP 10: ROC CURVE ==========
    print("\n" + "=" * 60)
    print("STEP 10: ROC Curve Analysis")
    print("=" * 60)
    
    # For multi-class, we'll create ROC curves for each class
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    y_pred_proba = best_model.predict_proba(X_test)
    
    plt.figure(figsize=(10, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, (label, color) in enumerate(zip(np.unique(y), colors)):
        if len(np.unique(y)) > 2:  # Multi-class
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = output_dir / "roc_curves.png"
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"ROC curve plot saved: {roc_plot_path}")

    # ========== STEP 11: CONFUSION MATRIX VISUALIZATION ==========
    print("\n" + "=" * 60)
    print("STEP 11: Confusion Matrix Visualization")
    print("=" * 60)
    
    plt.figure(figsize=(7, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y))
    plt.yticks(tick_marks, np.unique(y))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.tight_layout()
    cm_plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved: {cm_plot_path}")

    # ========== STEP 12: SAVE RESULTS ==========
    print("\n" + "=" * 60)
    print("STEP 12: Saving Results")
    print("=" * 60)
    
    results = {
        "problem": "Iris Species Classification",
        "dataset_info": {
            "total_samples": len(df),
            "features": list(feature_names),
            "classes": list(np.unique(y)),
            "class_distribution": dict(zip(np.unique(y), np.bincount(pd.Categorical(y, categories=np.unique(y)).codes)))
        },
        "preprocessing": {
            "method": "StandardScaler",
            "train_test_split": "80-20",
            "random_state": 42
        },
        "models_evaluated": {
            "Logistic Regression": {
                "test_accuracy": float(lr_acc),
                "cv_mean": float(cv_scores_lr.mean()),
                "cv_std": float(cv_scores_lr.std()),
            },
            "Random Forest": {
                "test_accuracy": float(rf_acc),
                "cv_mean": float(cv_scores_rf.mean()),
                "cv_std": float(cv_scores_rf.std()),
            },
            "SVM": {
                "test_accuracy": float(svm_acc),
                "cv_mean": float(cv_scores_svm.mean()),
                "cv_std": float(cv_scores_svm.std()),
            },
            "KNN": {
                "test_accuracy": float(knn_acc),
                "cv_mean": float(cv_scores_knn.mean()),
                "cv_std": float(cv_scores_knn.std()),
            }
        },
        "best_model": {
            "type": "Tuned Random Forest",
            "parameters": grid_search.best_params_,
            "cv_best_score": float(grid_search.best_score_),
            "test_accuracy": float(best_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "feature_importance": importance_df.to_dict(orient="records")
        }
    }
    
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # ========== STEP 13: MAKE PREDICTIONS ==========
    print("\n" + "=" * 60)
    print("STEP 13: Making Predictions on New Data")
    print("=" * 60)
    
    # Example: predict on first 5 test samples
    example_predictions = best_model.predict(X_test[:5])
    example_proba = best_model.predict_proba(X_test[:5])
    
    print("\nExample Predictions (first 5 test samples):")
    for i, (true_label, pred_label) in enumerate(zip(y_test[:5], example_predictions)):
        confidence = example_proba[i].max() * 100
        print(f"  Sample {i+1}: Predicted = {pred_label:12} | True = {true_label:12} | Confidence = {confidence:.2f}%")

    print("\n" + "=" * 60)
    print("Project Complete!")
    print("=" * 60)
    print(f"Results and visualizations saved to: {project_root}")
    print(f"Check the `plots/` folder for all generated charts.")


if __name__ == "__main__":
    main()
