# Food Delivery Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange?logo=apachespark)
![Databricks](https://img.shields.io/badge/Databricks-Serverless-red?logo=databricks)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

An end-to-end machine learning and deep learning project analysing 771,000 food delivery orders across Dubai, Abu Dhabi, Sharjah and Ajman to solve three real business problems using modern data science and AI techniques.


## What This Project Does

Food delivery platforms generate thousands of orders every day. Behind every order is a user who might stop ordering, a delivery that might go wrong, and a restaurant that might be quietly damaging the platform's reputation. This project uses machine learning and deep learning to detect all three problems before they escalate.

**Churn Prediction** Identifies users who are at risk of leaving the platform based on their order history, cancellation behaviour, and spending patterns. Built using XGBoost and LSTM to capture both static features and sequential ordering behaviour over time.

**Order Quality Risk Scoring** Scores every order based on how likely it is to result in a poor delivery experience. Factors include traffic conditions, delivery distance, driver availability, and restaurant performance. Built using XGBoost and Random Forest.

**Restaurant Health Scoring** Evaluates every restaurant on the platform based on their cancellation rate, average delivery time, and order quality. Flags underperforming restaurants before they drive users away. Built using XGBoost, Logistic Regression, SVM, and an XGBoost and SVM ensemble.


## Key Results

| Model | Problem | Accuracy | ROC AUC |
|-------|---------|----------|---------|
| XGBoost | Churn Prediction | 94.1% | 0.979 |
| LSTM (PyTorch) | Churn Prediction | 100.0%* | 1.000* |
| XGBoost | Order Quality Risk | 94.2% | 0.974 |
| Random Forest | Order Quality Risk | 93.3% | 0.967 |
| XGBoost | Restaurant Health | 97.2% | 0.993 |
| Logistic Regression | Restaurant Health | 95.0% | 0.995 |
| SVM | Restaurant Health | 92.8% | 0.992 |
| XGBoost + SVM Ensemble | Restaurant Health | 97.0% | 0.994 |

*LSTM achieved perfect results due to the inherently clean churn signal in synthetic data. See notebook 04 for full explanation.

## Model Visualisations

### XGBoost Churn Prediction — Confusion Matrix
![XGBoost Churn Confusion Matrix](assets/xgb_confusion_matrix.png)

Out of 5,000 test users, the XGBoost model correctly identified 4,643 non-churners and 1,004 churners. It raised 156 false alarms and missed 197 actual churners. The model is highly conservative with false alarms, making it practical for targeted retention campaigns where unnecessary outreach has a real cost.


### XGBoost Churn Prediction — SHAP Feature Importance
![XGBoost Churn SHAP](assets/xgb_shap_importance.png)

Total orders is by far the strongest predictor of churn, followed by total spend and Ramadan orders. Churners place significantly fewer orders overall and disengage during high-activity periods like Ramadan, which loyal users participate in heavily. City, payment method, and top cuisine had almost no influence on churn, confirming that where a user lives or what they order matters far less than how often and how consistently they order.


### LSTM Churn Prediction — Confusion Matrix
![LSTM Churn Confusion Matrix](assets/lstm_confusion_matrix.png)

The LSTM model achieved near-perfect results, correctly identifying 4,832 non-churners and 1,079 churners with zero false alarms and only 89 missed churners. These results reflect the inherently clean churn signal in synthetic data. See notebook 04 for a full explanation of this limitation.


### LSTM Churn Prediction — Training Loss Curve
![LSTM Training Loss](assets/lstm_training_loss.png)

The training loss drops sharply from 0.122 in epoch 1 to below 0.01 by epoch 3, then continues declining gradually to 0.0013 by epoch 20. This smooth downward curve confirms the LSTM trained stably and converged properly with no instability during training.


### XGBoost Order Quality Risk — Confusion Matrix
![XGBoost Order Risk Confusion Matrix](assets/xgb_order_risk_confusion_matrix.png)

Out of 154,247 test orders, the model correctly identified 127,958 low risk orders and 17,292 high risk orders. It raised 2,926 false alarms and missed 6,071 actual high risk orders. The model catches 74% of all genuinely risky orders, making it a reliable operational tool for pre-emptive delivery intervention.


### XGBoost Order Quality Risk — SHAP Feature Importance
![XGBoost Order Risk SHAP](assets/xgb_order_risk_shap.png)

Traffic level is the strongest predictor of order quality risk by a significant margin, followed by delivery duration and delivery distance. Restaurant health score also contributes meaningfully, confirming that poor restaurant performance increases delivery risk even before the driver picks up the order. Time of day, payment method, and weekend flag have almost no influence on delivery risk.


### Random Forest Order Quality Risk — Confusion Matrix
![Random Forest Order Risk Confusion Matrix](assets/rf_order_risk_confusion_matrix.png)

Random Forest correctly identified 129,177 low risk orders and 14,805 high risk orders. Compared to XGBoost, it raises fewer false alarms (1,707 vs 2,926) but misses significantly more actual high risk orders (8,558 vs 6,071). XGBoost is the stronger choice when catching risky orders is the priority.


### XGBoost Restaurant Health — Confusion Matrix
![XGBoost Restaurant Health Confusion Matrix](assets/xgb_restaurant_health_confusion_matrix.png)

Out of 400 test restaurants, the model correctly identified 301 healthy restaurants and 88 unhealthy ones. Only 7 false alarms and 4 missed detections across the entire test set. The model correctly flags 96% of underperforming restaurants while raising false alarms on only 2.3% of healthy ones.


### XGBoost Restaurant Health — SHAP Feature Importance
![XGBoost Restaurant Health SHAP](assets/xgb_restaurant_health_shap.png)

Average order quality risk dominates as the strongest predictor of restaurant health, followed by average delivery duration and cancellation rate. This confirms that restaurant underperformance is driven primarily by operational factors rather than cuisine type or location. Distinct items served has almost zero influence on health score.


## Tech Stack

- **Languages & Libraries:**
Python, PySpark 

- **Cloud & Platform:**
Databricks, Apache Spark

- **Machine Learning:**
XGBoost, Random Forest, Logistic Regression, SVM, Ensemble Methods

- **Deep Learning:**
LSTM (Long Short-Term Memory), PyTorch

- **Experiment Tracking:**
MLflow

- **Model Explainability:**
SHAP (SHapley Additive Explanations)

- **Dashboard & Visualisation:**
Streamlit, Matplotlib, Seaborn

- **Version Control:**
Git, GitHub

## Project Structure

```
food-delivery-analysis/
│
├── data/                          # Local data files (not tracked by Git)
│   └── uae_food_delivery_771k.csv # 771,000 row UAE food delivery dataset
│
├── scripts/                       # Data generation scripts
│   └── generate_uae_data.py       # Generates the synthetic UAE dataset
│
├── notebooks/                     # Databricks notebooks (in order of execution)
│   ├── 01_data_loading_and_ingestion.py   # Loads CSV and saves as Parquet
│   ├── 02_eda_and_data_quality.py         # Exploratory analysis and data validation
│   ├── 03_feature_engineering.py          # Builds feature tables for all three models
│   ├── 04_churn_prediction.py             # XGBoost and LSTM churn prediction models
│   ├── 05_order_quality_risk.py           # XGBoost and Random Forest risk scoring
│   ├── 06_restaurant_health_scoring.py    # XGBoost, Logistic Regression, SVM and Ensemble
│   └── 07_model_evaluation.py             # Cross model comparison (coming soon)
│
├── assets/                        # Charts and visualisations for README
│
├── dashboard/                     # Streamlit business intelligence dashboard (coming soon)
│
├── .gitignore                     # Excludes large data files from Git
└── README.md                      # Project documentation
```

## Dataset

771,000 synthetic food delivery orders generated from real UAE market statistics. Every distribution and pattern in the dataset is grounded in publicly documented UAE food delivery market data.

| Attribute | Detail |
|-----------|--------|
| Total Orders | 771,000 |
| Unique Users | 30,000 |
| Unique Restaurants | 2,000 |
| Unique Drivers | 3,000 |
| Cities | Dubai, Abu Dhabi, Sharjah, Ajman |
| Areas | 52 across all four cities |
| Cuisine Types | 12 (Indian, Arabic, Pakistani, American, Filipino, Chinese, Lebanese, Italian, Japanese, Thai, Mexican, Sri Lankan) |
| Date Range | January to December 2024 |
| Churn Rate | 11.9% at order level, 20% at user level |
| Average Order Value | 124 AED |

The dataset is generated locally using `scripts/generate_uae_data.py` and is not tracked by Git due to file size. Clone the repo and run the script to regenerate the exact same dataset using seed 42.


## Data Disclaimer

The dataset used in this project is synthetic. UAE food delivery platforms such as Talabat, Deliveroo, and Careem do not publish their operational data publicly. Rather than using an irrelevant dataset from a different market, the dataset was generated from scratch using real UAE market statistics including demographic breakdowns, cuisine preferences, city and area population distributions, and ordering behaviour patterns documented in publicly available market research.

The generation script is fully transparent, reproducible, and available in the `scripts/` folder. Running it with the fixed random seed of 42 produces the exact same 771,000 row dataset every time.

The churn signal in the dataset is inherently cleaner than real world data because it was designed at user creation time. In production, churn is driven by unpredictable human behaviour, competitive factors, and external events that no synthetic dataset can fully replicate. This limitation is documented transparently in notebook 04.


## Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| order_id | String | Unique identifier for each order |
| user_id | String | Unique identifier for each user |
| restaurant_id | String | Unique identifier for each restaurant |
| restaurant_name | String | Name and area of the restaurant |
| driver_id | String | Unique identifier for each driver |
| order_time | Timestamp | Date and time the order was placed |
| order_date | Date | Date the order was placed |
| order_hour | Integer | Hour of the day the order was placed (0 to 23) |
| is_weekend | Integer | 1 if the order was placed on Friday or Saturday, 0 otherwise |
| is_ramadan_period | Integer | 1 if the order was placed during Ramadan 2024, 0 otherwise |
| city | String | City where the order was delivered |
| area | String | Specific area within the city |
| cuisine | String | Cuisine type of the restaurant |
| item_name | String | Name of the item ordered |
| quantity | Integer | Number of items ordered |
| unit_price_aed | Float | Price per item in AED |
| delivery_fee_aed | Float | Delivery fee charged in AED |
| total_price_aed | Float | Total order value including delivery fee in AED |
| payment_method | String | Payment method used (Credit Card, Debit Card, Apple Pay, In-App Wallet) |
| delivery_distance_km | Float | Distance from restaurant to delivery address in kilometres |
| traffic_level | String | Traffic conditions at time of delivery (Low, Medium, High, Very High) |
| driver_vehicle | String | Vehicle type used for delivery (Motorcycle, Bicycle) |
| driver_availability | String | Driver availability status at time of order (Available, Busy, Unavailable) |
| delivery_duration_mins | Float | Actual delivery time in minutes |
| order_status | String | Final order status (Delivered, Cancelled, In Transit) |
| user_subscription | Integer | 1 if the user has an active subscription, 0 otherwise |
| restaurant_health_score | Float | Restaurant performance score between 0 and 1, higher is better |
| order_quality_risk_score | Float | Risk score for poor delivery experience between 0 and 1, higher means more risk |
| churn_risk | Integer | 1 if the user is at risk of churning, 0 otherwise |


## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/JasrajBhatia/food-delivery-analysis.git
cd food-delivery-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy faker scikit-learn xgboost shap torch mlflow streamlit
```

**3. Generate the dataset**
```bash
python3 scripts/generate_uae_data.py
```
This generates the full 771,000 row dataset in the `data/` folder. Takes approximately 3 to 5 minutes to complete.

**4. Run the notebooks**

Upload the dataset to your Databricks volume and run the notebooks in order from 01 through to 07. Each notebook loads from the Parquet files saved by the previous one.


## Status

Work in progress.

**Completed:**
- Data generation and ingestion
- Exploratory data analysis and data quality checks
- Feature engineering for all three models
- Churn prediction (XGBoost) — 94.1% accuracy, 0.979 ROC AUC
- Churn prediction (LSTM PyTorch) — 100.0% accuracy, 1.000 ROC AUC
- Order quality risk scoring (XGBoost) — 94.2% accuracy, 0.974 ROC AUC
- Order quality risk scoring (Random Forest) — 93.3% accuracy, 0.967 ROC AUC
- Restaurant health scoring (XGBoost) — 97.2% accuracy, 0.993 ROC AUC
- Restaurant health scoring (Logistic Regression) — 95.0% accuracy, 0.995 ROC AUC
- Restaurant health scoring (SVM) — 92.8% accuracy, 0.992 ROC AUC
- Restaurant health scoring (XGBoost + SVM Ensemble) — 97.0% accuracy, 0.994 ROC AUC

**Coming Soon:**
- Cross model evaluation notebook
- Streamlit dashboard
