# Beginner Data Science Project

This simple project teaches a beginner how to work with a dataset, inspect it with pandas, and create easy visualizations using matplotlib.

## What this project does

- Loads the Iris dataset from `scikit-learn`
- Converts it into a `pandas` DataFrame
- Shows the first rows and summary statistics
- Groups data by species and calculates average feature values
- Saves a CSV copy of the dataset
- Creates a histogram and a scatter plot

## Run the project

From the workspace root:

```bash
python iris_simple_analysis.py
```

## SVM classification and hyperparameter tuning

`iris_SVM_tuning.py` extends the basic analysis into a complete Iris flower
classification workflow. It:

- Loads and explores the Iris dataset
- Saves the dataset as `iris_dataset.csv`
- Creates histograms and feature scatter plots
- Scales the features with `StandardScaler`
- Splits the data into training and test sets
- Compares Logistic Regression, Random Forest, SVM, and K-Nearest Neighbors
- Evaluates each model with test accuracy and 5-fold cross-validation
- Tunes an SVM with `GridSearchCV` across multiple `C`, `gamma`, and `kernel` values
- Reports the tuned model's classification metrics and confusion matrix
- Creates ROC curve, confusion matrix, and linear-SVM feature-importance plots when applicable
- Saves the best model parameters and evaluation results as JSON

Run it from the workspace root:

```bash
python iris_SVM_tuning.py
```

The script creates `svm_tuning_results.json` and saves generated charts in
`svm_tuning_plots/`.

## Files

- `iris_simple_analysis.py` — main script with comments for beginners
- `iris_SVM_tuning.py` — compares classification models and tunes an SVM
- `requirements.txt` — required packages
- `iris_dataset.csv` — saved dataset output (created when the script runs)
- `plots/` — generated charts saved by the script
- `svm_tuning_results.json` — model comparison and tuned SVM results (created when the tuning script runs)
- `svm_tuning_plots/` — charts created by the tuning script
