# Mushroom Classification

Classification of mushrooms as edible or poisonous using five machine learning models and a Streamlit application.

## a. Problem statement

Mushroom classification is a binary classification problem in which the objective is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

The dataset contains categorical attributes such as cap shape, cap surface, cap color, odor, gill characteristics, stalk characteristics and habitat. The goal is to identify patterns in these attributes and classify mushrooms into their respective categories.

I implemented five machine learning classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) using the same training and testing dataset and compared their performance using six evaluation metrics.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository – Mushroom Dataset (verify that this is your actual download source) |
| Instances | 8,124 |
| Original Features | 22 categorical input features |
| Usable Features | 21 after removing the constant `veil-type` feature |
| Target | `class` – two classes: edible (`e`) and poisonous (`p`) |
| Class Counts | Edible: 4,208; Poisonous: 3,916 |

The dataset contains categorical mushroom characteristics. The missing values represented by `?` were converted into NaN values and handled using most-frequent imputation. The constant `veil-type` feature was removed because it does not provide predictive information.

All categorical input features were transformed using One-Hot Encoding. The dataset was divided into training and testing sets using an 80:20 stratified split with `random_state=42`.

The preprocessing steps were incorporated into the machine learning pipelines to prevent data leakage. The held-out test dataset was exported as `test_data.csv` for evaluation through the Streamlit application.

## c. GitHub Repository Link
[View Mushroom Classification GitHub Repository](https://github.com/kovvururajesh7/mushroom-classification)


**Files in the repository:**

- `Mushroom_Classification.ipynb` – Notebook containing dataset preprocessing, model training and evaluation.
- `app.py` – Streamlit application that loads the saved models and evaluates uploaded test data without retraining.
- `requirements.txt` – Required Python dependencies.
- `test.csv` – Held-out test dataset containing input features and the target column.
- `train.csv` – Original mushroom dataset.
- `model_comparison_metrics.csv` – Evaluation metrics for all five classifiers.
- `model/` – Contains all five saved Joblib model pipelines and `feature_schema.json`.

To run the application locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## d. Models used

Results on the held-out test dataset (1,625 test instances).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9988 | 1.0000 | 1.0000 | 0.9974 | 0.9987 | 0.9975 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9569 | 0.9963 | 0.9199 | 0.9974 | 0.9571 | 0.9170 |
| Support Vector Machine | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Observations**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 99.88% accuracy and 99.87% F1 score, demonstrating excellent classification performance. Its high precision and recall indicate that it effectively distinguishes between edible and poisonous mushrooms. |
| Decision Tree | Achieved perfect scores across all six evaluation metrics. It effectively captured the relationships between mushroom characteristics and their respective classes on the test dataset. |
| KNN | Achieved perfect classification scores across all six metrics. This indicates that samples with similar mushroom characteristics were effectively classified into their respective categories. |
| Naive Bayes | Achieved 95.69% accuracy and 95.71% F1 score. Its recall was high at 99.74%, but its comparatively lower precision of 91.99% indicates that some edible mushrooms were incorrectly classified as poisonous. |
| Support Vector Machine | Achieved perfect scores across all six evaluation metrics. Its RBF kernel effectively captured the classification boundaries between edible and poisonous mushrooms. |

**Overall winner**

Decision Tree, K-Nearest Neighbors and Support Vector Machine achieved identical performance, with 100% scores across all six evaluation metrics. Therefore, these three models are considered joint top-performing models on the held-out test dataset.

## e. Streamlit app

The Streamlit application provides an interactive interface for evaluating the five trained mushroom classification models.

- **Sidebar:** Upload a CSV test dataset and select one of the five trained classification models using the dropdown.
- **Evaluation:** Displays Accuracy, AUC, Precision, Recall, F1 Score and MCC for the selected model, along with its confusion matrix and classification report.
- **Model comparison:** Displays a comparison table of all five classifiers using the six evaluation metrics and a performance comparison chart.
- **Prediction download:** Provides an option to download the predictions generated by the selected model.

The uploaded CSV must contain all 21 required input features and the target column `class`, with target values `e` and `p`. The application validates the uploaded dataset and displays the evaluation results without retraining the models.

The application loads the previously trained Joblib model pipelines and their preprocessing artifacts from the `model/` directory. The bundled `test_data.csv` can also be used for evaluation.

**Live Streamlit application:** 
