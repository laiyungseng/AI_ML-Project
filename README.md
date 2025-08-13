# AI_ML-Project

Welcome to my **AI/ML Learning Portfolio**!  
This repository documents my structured journey into **Artificial Intelligence** and **Machine Learning**, with a strong focus on **Data Analysis**, **Machine Learning Models**, **Time-Series Forecasting**, and **Model Optimization**.

---

## 📌 Overview
This repository features:

- **Data Analysis** → Exploratory Data Analysis (EDA), feature engineering, and preprocessing.
- **Machine Learning Models** → Regression, classification, and forecasting using tree-based and deep learning models.
- **Time-Series Forecasting** → ARIMA, Prophet, LSTM, and Transformer-based approaches.
- **Model Optimization** → Hyperparameter tuning with Optuna and performance benchmarking.
- **Project-Based Learning** → Each month contains a focused mini-project with practical datasets.

---

## 📂 Repository Structure
notebooks/ → Jupyter Notebooks for analysis, modeling, and forecasting
data/ → Sample datasets (if included)
models/ → Saved ML/DL models
README.md → Project overview and roadmap

markdown
Copy
Edit

---

## 📅 Monthly Progress

### **Month 1 – Data Analysis**
**Goal**: Learn dataset exploration, cleaning, and visualization.  
**Project**: Electricity Consumption Data Analysis.  
**Highlights**:
- Data cleaning & preprocessing (missing values, outliers)
- EDA with Matplotlib & Seaborn
- Trend insights in energy consumption

**Skills**: `pandas`, `matplotlib`, `seaborn`  
📄 Notebook: `Electricity_consumption_Data_Analysis_tutorial.ipynb`

---

### **Month 2 – Machine Learning Models**
**Goal**: Implement and compare ML forecasting models.  
**Project**: Electricity Load Forecasting (Random Forest vs Baseline).  
**Highlights**:
- Feature selection via correlation analysis
- Models: Random Forest, Linear Regression (baseline), XGBoost
- Metrics: RMSE, MAE, R²

**Skills**: `scikit-learn`, `xgboost`, feature engineering  
📄 Notebook: `DataPrediction_Tutorial2.ipynb`

---

### **Month 3 – Time-Series Forecasting with Deep Learning**
**Goal**: Build sequence-based forecasting models.  
**Project**: Electricity Price Forecasting (LSTM vs XGBoost).  
**Highlights**:
- Created lag & rolling mean features
- Implemented XGBoost, Random Forest, and LSTM
- LSTM excelled in capturing temporal dependencies

**Skills**: `tensorflow/keras`, `xgboost`, time-series modeling  
📄 Notebook: `Tutorial_3_Deep_Learning.ipynb`

---

### **Month 4 – Transformer-Based Time-Series Forecasting**
**Goal**: Apply advanced architectures & hyperparameter tuning.  
**Project**: Electricity Price Forecasting with Temporal Fusion Transformer (TFT).  
**Highlights**:
- Built TFT model for multi-horizon forecasting
- Hyperparameter tuning with Optuna
- Compared TFT vs LSTM vs XGBoost on RMSE, MAE, MAPE
- TFT achieved best overall balance between accuracy and generalization

**Skills**: `pytorch`, `pytorch-forecasting`, `optuna`, transformer architecture  
📄 Notebook: `Month4_TFT_Optimization.ipynb`  

**Metrics Summary**:

| Model   | RMSE   | MAE    | MAPE (%) | R²   |
|---------|--------|--------|----------|------|
| XGBoost | 12.34  | 8.76   | 4.12     | 0.89 |
| LSTM    | 11.98  | 8.54   | 4.05     | 0.90 |
| TFT     | 10.85  | 7.92   | 3.87     | 0.92 |

---

## 🚀 Roadmap

- **Month 5** → Model Deployment & Interactive Visualization  
  - Build Streamlit dashboard for predictions  
  - Create FastAPI service for REST-based model access  
  - Containerize with Docker for easy deployment  

- **Month 6** → Real-time Data Ingestion & Forecasting  
  - Integrate live energy price feeds  
  - Real-time inference & dashboard updates  

- **Month 7** → End-to-End ML Pipeline with MLOps  
  - Model retraining automation  
  - CI/CD for ML models with GitHub Actions  

---

## 🛠 Technologies Used
- **Languages**: Python
- **Core Libraries**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow`, `pytorch`
- **Time-Series Tools**: `statsmodels`, `prophet`, `pytorch-forecasting`
- **Optimization**: `optuna`
- **Deployment**: `streamlit`, `fastapi`, `docker`

---

## 📥 Getting Started
```bash
git clone https://github.com/laiyungseng/AI_ML-Project.git
cd AI_ML-Project
