# Heart Disease Prediction from Medical Data

An educational supervised-learning project that compares classification models on the UCI Cleveland Heart Disease dataset. The notebook uses 13 clinical attributes to predict the dataset's binary heart-disease label.

> **Important:** This is an academic machine-learning exercise, not a medical device or diagnostic tool. Its predictions must not be used for clinical decisions.

## Project Files

| File | Description |
| --- | --- |
| `Disease Prediciton.ipynb` | Notebook for loading, preprocessing, training, evaluation, and visualisation |
| `heart+disease/` | Local copy of the Cleveland dataset and related UCI reference files |

## Workflow

1. Load the Cleveland dataset and treat `?` as missing values.
2. Remove incomplete rows and convert the original target to a binary label.
3. Explore the data and split it into stratified training and test sets.
4. Fit a `StandardScaler` on the training data and transform both splits.
5. Train Logistic Regression, Random Forest, SVM, and XGBoost classifiers.
6. Compare accuracy and ROC-AUC, then inspect the selected model's classification report, confusion matrix, and ROC curve.

## Dataset

The notebook currently downloads `processed.cleveland.data` from the UCI Machine Learning Repository when it runs. A local copy is also included under `heart+disease/`.

For an offline run, change the dataset-loading path in the notebook to:

```python
df = pd.read_csv(
    "heart+disease/processed.cleveland.data",
    names=column_names,
    na_values="?"
)
```

After removing rows with missing values, the notebook works with 297 records and 13 input features.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Run Locally

Install the dependencies:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `Disease Prediciton.ipynb` and run the cells in order. An internet connection is required with the notebook's current UCI download URL; use the local path above for offline execution.

## Limitations

- The dataset is small and historical.
- Model results depend on the train/test split and software versions.
- The project does not include external clinical validation.
- It is intended for learning classification workflows, not for diagnosing or treating disease.
