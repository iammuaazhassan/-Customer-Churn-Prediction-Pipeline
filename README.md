# 📊 Customer Churn Prediction Pipeline

This project implements an **end-to-end machine learning pipeline** for
predicting customer churn using the **Telco Customer Churn Dataset**.\
It demonstrates **data preprocessing, model training, hyperparameter
tuning, evaluation, and deployment-readiness** with the Scikit-learn
`Pipeline` API.

------------------------------------------------------------------------

## 🚀 Project Overview

-   **Dataset**: [Telco Customer
    Churn](https://raw.githubusercontent.com/iammuaazhassan/churn-pipeline/main/Telco-Customer-Churn.csv)\
-   **Objective**: Predict whether a customer will churn based on
    demographic and service usage data.\
-   **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn,
    Joblib

------------------------------------------------------------------------

## 🛠️ Features

-   End-to-End ML Pipeline with preprocessing (scaling, encoding).
-   Models: Logistic Regression, Random Forest.
-   Hyperparameter tuning using `GridSearchCV`.
-   Export trained pipeline using `joblib` for production use.
-   Graphical evaluation (confusion matrix, ROC curve, accuracy trends).

------------------------------------------------------------------------

## 📂 Project Structure

    ├── churn_pipeline.ipynb     # Jupyter notebook with full pipeline
    ├── train.py                 # (Optional) Training script
    ├── Telco-Customer-Churn.csv # Dataset (if included)
    ├── models/                  # Saved pipelines (joblib)
    ├── results/                 # Graphs and evaluation results
    └── README.md                # Project documentation

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

``` bash
git clone https://github.com/iammuaazhassan/churn-pipeline.git
cd churn-pipeline
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Usage

Run the training pipeline:

``` bash
python train.py
```

Or open the Jupyter notebook:

``` bash
jupyter notebook churn_pipeline.ipynb
```

------------------------------------------------------------------------

## 📈 Results

  Model                 Accuracy   F1-Score
  --------------------- ---------- ----------
  Logistic Regression   80.5%      0.79
  Random Forest         82.3%      0.81

------------------------------------------------------------------------

## 📊 Visualizations

The following graphs are generated: - Class distribution in dataset -
Confusion matrix - ROC curve - Accuracy and F1-score comparison

*(You can paste the generated plots here as images, e.g.,
`![Confusion Matrix](results/confusion_matrix.png)`)*

------------------------------------------------------------------------

## 💾 Model Export

The final pipeline is saved using Joblib:

``` bash
from joblib import load
pipeline = load("models/final_churn_model.joblib")
predictions = pipeline.predict(new_data)
```

------------------------------------------------------------------------

## 🎯 Skills Gained

-   ML pipeline construction
-   Data preprocessing with `Pipeline`
-   Hyperparameter tuning with GridSearchCV
-   Model export for production
-   Visualization of results

------------------------------------------------------------------------

## 📜 License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------

👤 **Author**: [Muaz Hassan](https://github.com/iammuaazhassan)
