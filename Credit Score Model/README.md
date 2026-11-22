# Heart Disease Prediction from Medical Data

A supervised machine learning classification project that predicts heart disease presence from clinical and diagnostic patient attributes using the UCI Cleveland Heart Disease dataset (`297` cleaned samples and `13` features from `processed.cleveland.data` and supporting files under `heart+disease/`) inside `Disease Prediciton.ipynb`.

## Project Structure & Data Files
- `Disease Prediciton.ipynb`: End-to-end Jupyter notebook covering dataset loading, missing-value handling (`?` to `NaN` and row removal), binary target conversion, exploratory statistics, stratified train/test splitting (`80/20`, `60` test samples), feature standardization (`StandardScaler`), multi-model training and evaluation, diagnostic visualizations, sample predictions, and summary reporting.
- `heart+disease/`: Local UCI Heart Disease archive directory containing `processed.cleveland.data`, `cleveland.data`, `hungarian.data`, `switzerland.data`, `long-beach-va.data`, `heart-disease.names`, and reference/cost files.

## Dataset Features & Target
- **Clinical & Diagnostic Features (13):** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar), `restecg` (resting electrocardiographic results), `thalach` (maximum heart rate achieved), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope` (slope of peak exercise ST segment), `ca` (number of major vessels colored by fluoroscopy), and `thal` (thalassemia).
- **Binary Target (`target`):** Raw multi-level target values (`> 0`) are converted into a binary classification target (`0` = No Disease, `1` = Disease).

## Detailed Workflow
1. **Loading & Cleaning:** Reads the 14-column Cleveland dataset (treating `?` as missing values), inspects initial shape and missing value counts, drops incomplete rows (`df.dropna()`) leaving `297` records, and binarizes `target`.
2. **Exploratory Analysis & Splitting:** Reviews sample rows and descriptive statistics, separates features (`X`, `13` columns) and target (`y`), and performs an `80/20` stratified split (`random_state=42`).
3. **Feature Scaling:** Standardizes features using `StandardScaler` (`fit_transform` on `X_train` and `transform` on `X_test`).
4. **Multi-Model Training & Comparison:** Trains `Logistic Regression`, `Random Forest` (`n_estimators=100`, `random_state=42`), `SVM` (`probability=True`, `random_state=42`), and `XGBoost` (`XGBClassifier`, `random_state=42`), recording `accuracy` and `ROC-AUC` for each classifier and selecting the top model by accuracy.
5. **Visualizations & Sample Inference:** Plots the best model's confusion matrix, ROC curve, feature importance or model accuracy comparison, and actual vs. predicted points, then demonstrates per-sample predictions and predicted probabilities on test instances.

## Model Performance Summary
- **Logistic Regression:** `Accuracy = 0.8333`, `AUC = 0.9498`
- **Random Forest (Top Accuracy):** `Accuracy = 0.8667`, `AUC = 0.9414`
- **SVM:** `Accuracy = 0.8500`, `AUC = 0.9542`
- **XGBoost:** `Accuracy = 0.8667`, `AUC = 0.8917`

## Tools & Libraries
- **Python**
- **Data & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `StandardScaler`, `train_test_split`, `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`) and `xgboost` (`XGBClassifier`)

## How to Run
1. Open `Disease Prediciton.ipynb` in **Jupyter Notebook**, **Google Colab**, or **VS Code** with `xgboost` installed alongside `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.
2. Run all cells sequentially from top to bottom (update the dataset source to `heart+disease/processed.cleveland.data` if running in an offline environment).
