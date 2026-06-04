# Food Delivery Analysis

An end-to-end machine learning project analysing 750,000 food delivery orders across Dubai, Abu Dhabi, Sharjah and Ajman to solve three real business problems.

## What This Project Does

Food delivery platforms generate thousands of orders every day. 
Behind every order is a user who might stop ordering, a delivery 
that might go wrong, and a restaurant that might be quietly 
damaging the platform's reputation. This project uses machine 
learning to detect all three problems before they escalate.

**Churn Prediction**
Identifies users who are at risk of leaving the platform based 
on their order history, cancellation behaviour, and spending 
patterns. Built using XGBoost and LSTM to capture both static 
features and sequential ordering behaviour over time.

**Order Quality Risk Scoring**
Scores every order in real time based on how likely it is to 
result in a poor delivery experience. Factors include traffic 
conditions, delivery distance, driver availability, and 
restaurant performance. Built using XGBoost and Random Forest.

**Restaurant Health Scoring**
Evaluates every restaurant on the platform based on their 
cancellation rate, average delivery time, and order quality. 
Flags underperforming restaurants before they drive users away. 
Built using XGBoost.

## Tech Stack

**Languages & Libraries**
Python, PySpark

**Cloud & Platform**
Databricks, Apache Spark

**Machine Learning**
XGBoost, Random Forest, Scikit-learn

**Deep Learning**
LSTM (Long Short-Term Memory), TensorFlow/Keras

**Experiment Tracking**
MLflow

**Model Explainability**
SHAP (SHapley Additive Explanations)

**Dashboard & Visualisation**
Streamlit, Matplotlib, Seaborn

**Version Control**
Git, GitHub

## Project Structure

```
food-delivery-analysis/
│
├── data/                          # Local data files (not tracked by Git)
│   └── uae_food_delivery_750k.csv # 750,000 row UAE food delivery dataset
│
├── scripts/                       # Data generation scripts
│   └── generate_uae_data.py       # Generates the synthetic UAE dataset
│
├── notebooks/                     # Databricks notebooks (in order of execution)
│   ├── 01_data_loading_and_ingestion.py   # Loads CSV and saves as Parquet
│   ├── 02_eda_and_data_quality.py         # Exploratory analysis and data validation
│   ├── 03_feature_engineering.py          # Builds feature tables for all three models
│   ├── 04_churn_prediction.py             # XGBoost and LSTM churn model (coming soon)
│   ├── 05_order_quality_risk.py           # XGBoost and Random Forest risk model (coming soon)
│   ├── 06_restaurant_health_scoring.py    # XGBoost restaurant health model (coming soon)
│   └── 07_model_evaluation.py             # Model comparison and SHAP explainability (coming soon)
│
├── dashboard/                     # Streamlit business intelligence dashboard (coming soon)
│
├── .gitignore                     # Excludes large data files from Git
└── README.md                      # Project documentation
```

## Dataset

750,000 synthetic food delivery orders generated from real UAE market statistics. Covers 4 cities, 52 areas, 12 cuisine types, and a full year of ordering behaviour (Jan to Dec 2024).

## Status

Work in progress. Feature engineering complete. Model training in progress.
