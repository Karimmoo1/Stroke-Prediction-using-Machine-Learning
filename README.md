# Stroke Prediction using Machine Learning

This project analyzes a medical dataset to predict the likelihood of a patient having a stroke using various machine learning classifiers. It includes data preprocessing, feature engineering, model training, evaluation, and comparison.

## 📁 Dataset

The dataset used is the **[Stroke Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)**.

You can download it directly from the Kaggle link above.

### 📊 Features

| Feature              | Description                                 |
|----------------------|---------------------------------------------|
| `id`                 | Unique identifier (dropped in preprocessing) |
| `gender`             | Gender of the patient                       |
| `age`                | Age of the patient                          |
| `hypertension`       | 1 if the patient has hypertension, else 0   |
| `heart_disease`      | 1 if the patient has any heart disease, else 0 |
| `ever_married`       | Marital status                              |
| `work_type`          | Type of work (e.g. Private, Self-employed)  |
| `Residence_type`     | Urban or Rural                              |
| `avg_glucose_level`  | Average glucose level in blood              |
| `bmi`                | Body Mass Index (missing values filled)     |
| `smoking_status`     | Smoking status                              |
| `stroke`             | Target variable (1 = stroke, 0 = no stroke) |

---

## 🔧 Preprocessing Steps

- Dropped `id` column.
- Imputed missing values in `bmi` using the median.
- Label encoded binary columns: `gender`, `ever_married`, `Residence_type`.
- One-hot encoded: `work_type`, `smoking_status`.
- Standard scaled: `age`, `avg_glucose_level`, `bmi`.

---

## 🤖 Models Implemented

- Logistic Regression
- SGD Classifier
- Perceptron
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)
- Linear Support Vector Classifier (SVC)
- Multi-Layer Perceptron (MLPClassifier)

> **Note**: The dataset is imbalanced. **SMOTE (Synthetic Minority Over-sampling Technique)** was used to oversample the minority class during training.

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate the models:
- Accuracy
- Precision
- Recall
- F1 Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Confusion Matrix
- Classification Report




