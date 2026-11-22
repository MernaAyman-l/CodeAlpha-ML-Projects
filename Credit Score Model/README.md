# Credit Score Model — Credit Risk Classification (CodeAlpha Task 1)

A machine learning classification project that constructs a binary creditworthiness target from historical monthly credit payment records and predicts applicant credit risk using `application_record.csv` and `credit_record.csv` inside `Task1 Credit Score Model.ipynb`.

## Project Structure & Artifacts
- `Task1 Credit Score Model.ipynb`: Complete workflow covering data loading, target creation from credit payment status history, inner merge on `ID`, statistical and visual EDA, feature cleaning and encoding, stratified train/test splitting, model training and comparison, detailed evaluation, feature importance analysis, and CSV export.
- `application_record.csv`: Applicant demographic, financial, household, and property attributes (`ID`, income, employment and birth days, family status, education, income type, occupation, realty/car ownership, contact flags).
- `credit_record.csv`: Monthly account status history (`STATUS` values such as `C`, `X`, `0`–`5`) tracked per applicant `ID`.
- `model_comparison.csv`: Evaluation comparison across trained models (`Precision`, `Recall`, `F1-Score`, `ROC-AUC`).
- `feature_importance.csv`: Ranked feature importance scores showing top predictors (`AGE`, `EMPLOYMENT_YEARS`, `AMT_INCOME_TOTAL`, `NAME_FAMILY_STATUS`, `FLAG_OWN_REALTY`, and others).
- `credit_scoring_predictions.csv`: Test set features alongside `ACTUAL_TARGET`, `PREDICTED_TARGET`, and `PREDICTION_PROBABILITY`.

## Detailed Methodology
1. **Data Loading & Initial Inspection:** Loads `application_record.csv` and `credit_record.csv`, inspects dataset dimensions, schemas, sample rows, and monthly `STATUS` distributions.
2. **Target Engineering (`create_credit_target`):** Aggregates records by `ID` to identify record counts and worst payment status, mapping non-severe/current statuses (`C`, `X`, `0`) to `TARGET = 1` (Good Credit) and severe delinquency statuses (`1`–`5`) to `TARGET = 0` (Bad Credit).
3. **Dataset Merging & EDA:** Merges `application_record.csv` with the engineered target table via `ID`, checks missing values, reviews numerical summaries, and visualizes target distribution alongside income, age, income type, education, and family status breakdowns.
4. **Preprocessing & Feature Preparation:** Drops identifier, raw day/record helper, and low-value flag columns (`ID`, `FLAG_MOBIL`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `worst_status`, `record_count`), encodes categorical variables with `LabelEncoder`, and separates `X` and `y`.
5. **Model Training & Evaluation:** Splits data `80/20` with `stratify=y` (`random_state=42`), scales features with `StandardScaler` (for `Logistic Regression`), and trains `Logistic Regression`, `Decision Tree`, and `Random Forest` (`n_estimators=100`, `random_state=42`), evaluating each model with `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, confusion matrix, ROC curve, classification report, and feature importance.
6. **Result Persistence:** Exports `credit_scoring_predictions.csv`, `model_comparison.csv`, and `feature_importance.csv`.

## Tools & Libraries
- **Python**
- **Data Manipulation & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning & Evaluation:** `scikit-learn` (`train_test_split`, `StandardScaler`, `LabelEncoder`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`, `precision_score`, `recall_score`, `f1_score`)

## How to Run
1. Place `application_record.csv` and `credit_record.csv` inside the `Credit Score Model/` folder.
2. Open `Task1 Credit Score Model.ipynb` in **Jupyter Notebook**, **Google Colab**, or **VS Code**.
3. Execute all cells sequentially from top to bottom to reproduce training, visualizations, and CSV outputs.
```[cite: 8]

---

### 2. Disease Prediction from Medical Data (`README.md`)

```markdown
# Heart Disease Prediction from Medical Data

A supervised machine learning classification project that predicts heart disease presence from clinical and diagnostic patient attributes using the UCI Cleveland Heart Disease dataset (`processed.cleveland.data` and supporting files under `heart+disease/`) inside `Disease Prediciton.ipynb`.

## Project Structure & Data Files
- `Disease Prediciton.ipynb`: End-to-end Jupyter notebook covering dataset loading, missing-value handling (`?` to `NaN` and row removal), binary target conversion, exploratory statistics, stratified train/test splitting, feature standardization (`StandardScaler`), multi-model training and evaluation, diagnostic visualizations, sample predictions, and summary reporting.
- `heart+disease/`: Local UCI Heart Disease archive directory containing `processed.cleveland.data`, `cleveland.data`, `hungarian.data`, `switzerland.data`, `long-beach-va.data`, `heart-disease.names`, and reference/cost files.

## Dataset Features & Target
- **Clinical & Diagnostic Features (13):** `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar), `restecg` (resting electrocardiographic results), `thalach` (maximum heart rate achieved), `exang` (exercise-induced angina), `oldpeak` (ST depression), `slope` (slope of peak exercise ST segment), `ca` (number of major vessels colored by fluoroscopy), and `thal` (thalassemia).
- **Binary Target (`target`):** Raw multi-level target values (`> 0`) are converted into a binary classification target (`0` = No Disease, `1` = Disease).

## Detailed Workflow
1. **Loading & Cleaning:** Reads the 14-column Cleveland dataset (treating `?` as missing values), inspects initial shape and missing value counts, drops incomplete rows (`df.dropna()`), and binarizes `target`.
2. **Exploratory Analysis & Splitting:** Reviews sample rows and descriptive statistics, separates features (`X`) and target (`y`), and performs an `80/20` stratified split (`random_state=42`).
3. **Feature Scaling:** Standardizes features using `StandardScaler` (`fit_transform` on `X_train` and `transform` on `X_test`).
4. **Multi-Model Training & Comparison:** Trains `Logistic Regression`, `Random Forest` (`n_estimators=100`, `random_state=42`), `SVM` (`probability=True`, `random_state=42`), and `XGBoost` (`XGBClassifier`, `random_state=42`), recording `accuracy` and `ROC-AUC` for each classifier and selecting the top model by accuracy.
5. **Visualizations & Sample Inference:** Plots the best model's confusion matrix, ROC curve, feature importance or model accuracy comparison, and actual vs. predicted points, then demonstrates per-sample predictions and predicted probabilities on test instances.

## Tools & Libraries
- **Python**
- **Data & Visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `SVC`, `StandardScaler`, `train_test_split`, `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`) and `xgboost` (`XGBClassifier`)

## How to Run
1. Open `Disease Prediciton.ipynb` in **Jupyter Notebook**, **Google Colab**, or **VS Code** with `xgboost` installed alongside `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.
2. Run all cells sequentially from top to bottom (update the dataset source to `heart+disease/processed.cleveland.data` if running in an offline environment).
```[cite: 8]

---

### 3. Handwritten Character Recognition (`README.md`)

```markdown
# Handwritten Character Recognition — MNIST CNN Classifier (CodeAlpha Task 3)

A deep learning computer vision project that classifies `28x28` grayscale handwritten digit images (`0`–`9`) from the `MNIST` dataset using a Convolutional Neural Network (`CNN`) built with TensorFlow/Keras in `Task 3 Handwritten Character Recognition.ipynb`.

## Project Structure
- `Task 3 Handwritten Character Recognition.ipynb`: Complete notebook covering library imports, MNIST dataset loading and inspection, pixel normalization and tensor reshaping, one-hot label encoding, CNN architecture construction and compilation, model training with validation split, test evaluation, classification report and per-digit accuracy breakdown, diagnostic plots, and performance summary.

## CNN Architecture & Training Setup
- **Input Preprocessing:** Scales pixel intensities from `[0, 255]` to `[0.0, 1.0]` (`float32`), reshapes images to `(28, 28, 1)` for single-channel 2D convolutions, and one-hot encodes labels across `10` digit classes (`to_categorical(..., 10)`).
- **Network Layers (`Sequential`):**
  - `Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))` + `MaxPooling2D((2, 2))`
  - `Conv2D(64, (3, 3), activation='relu')` + `MaxPooling2D((2, 2))`
  - `Conv2D(64, (3, 3), activation='relu')`
  - `Flatten()` -> `Dense(64, activation='relu')` -> `Dropout(0.5)` -> `Dense(10, activation='softmax')`
- **Compilation & Training:** Compiled with `optimizer='adam'`, `loss='categorical_crossentropy'`, and `metrics=['accuracy']`, then trained for `10` epochs with `batch_size=128` and `validation_split=0.2`.

## Evaluation & Visualizations
- Evaluates overall `test_loss` and `test_accuracy` on the MNIST test set and generates class predictions via `np.argmax(model.predict(x_test), axis=1)`.
- Computes `classification_report`, weighted `precision`, `recall`, `f1_score`, and accuracy per digit (`0`–`9`).
- Plots training vs. validation accuracy and loss curves over epochs, sample/misclassified digit images, and per-digit accuracy bars.

## Tools & Libraries
- **Python**
- **Deep Learning:** `tensorflow` / `keras` (`mnist`, `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `to_categorical`)
- **Data, Metrics & Visualization:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (`classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`)

## How to Run
1. Open `Task 3 Handwritten Character Recognition.ipynb` in **Jupyter Notebook**, **Google Colab**, or **VS Code** with `tensorflow` installed.
2. Run all cells sequentially from top to bottom (the notebook loads `tensorflow.keras.datasets.mnist` and trains the CNN from scratch).
```[cite: 8]

لو حابة نزود أي أقسام تانية زي متطلبات التشغيل (`requirements.txt`) أو شرح تفصيلي لأي كود جوه النوتبوكس قوليلي[cite: 8].
