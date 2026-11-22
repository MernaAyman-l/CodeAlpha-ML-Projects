# Handwritten Character Recognition — MNIST CNN Classifier (CodeAlpha Task 3)

A deep learning computer vision project that classifies `28x28` grayscale handwritten digit images (`0`–`9`) from the `MNIST` dataset (`10,000` test samples) using a Convolutional Neural Network (`CNN`) built with TensorFlow/Keras in `Task 3 Handwritten Character Recognition.ipynb`.

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
- Achieves `99.23%` test accuracy (`0.9923` accuracy, `0.0290` test loss, weighted `Precision = 0.9923`, `Recall = 0.9923`, `F1-Score = 0.9923`, with `9,923` correct out of `10,000` test images and per-digit accuracy ranging from `0.9871` on digit `9` to `0.9965` on digit `1`).
- Generates `classification_report`, weighted `precision`, `recall`, `f1_score`, and accuracy per digit (`0`–`9`).
- Plots training vs. validation accuracy and loss curves over epochs, sample/misclassified digit images, and per-digit accuracy bars.

## Tools & Libraries
- **Python**
- **Deep Learning:** `tensorflow` / `keras` (`mnist`, `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `to_categorical`)
- **Data, Metrics & Visualization:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (`classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`)

## How to Run
1. Open `Task 3 Handwritten Character Recognition.ipynb` in **Jupyter Notebook**, **Google Colab**, or **VS Code** with `tensorflow` installed.
2. Run all cells sequentially from top to bottom (the notebook loads `tensorflow.keras.datasets.mnist` and trains the CNN from scratch).
```[cite: 8]

لو حابة نضيف أي أقسام تانية زي `requirements.txt` أو نعدّل أي جزئية في أي واحد منهم قوليلي.
