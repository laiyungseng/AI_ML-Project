# Tutorial3
Deep learning (LSTM)
# **Month 3 Project Completion Report – Time-Series Forecasting**

## **Project Overview**
The goal of Month 3 was to advance from traditional ML models to **time-series forecasting using deep learning techniques**. The project focused on predicting electricity prices by incorporating temporal dependencies through **lag features, classical ML models, and LSTM deep learning networks**.

---

## **Key Achievements**
### **1. Data Preparation & EDA**
- Created lag features (`price_lag1`, `price_lag24`, rolling means) to capture temporal patterns.
- Conducted **ACF/PACF analysis** to identify autocorrelation and seasonality.
- Visualized price trends, highlighting daily and weekly fluctuations.

### **2. Model Development**
- Implemented and compared multiple models:
  - **XGBoost** for feature-based time-series forecasting.
  - **Random Forest** as an additional ML benchmark.
  - **LSTM (Long Short-Term Memory)** deep learning model for sequence forecasting.
- Applied **time-series train-test split** to prevent data leakage.

### **3. Model Evaluation**
- Metrics used: **RMSE, MAE, R², and MAPE**.
- Developed a **model comparison table** summarizing performance across:
  - Baseline model
  - XGBoost
  - LSTM (best-performing deep learning model)

### **4. Final Insights**
- **LSTM outperformed other models**, effectively capturing long-term dependencies.
- XGBoost was strong for short-term prediction but lacked temporal memory.
- Lag features and rolling statistics significantly improved ML performance.

---

## **Next Steps**
- **Hyperparameter tuning:** Use GridSearchCV or Optuna for both XGBoost and LSTM optimization.
- **Hybrid approaches:** Explore combining ML with deep learning (e.g., XGBoost + LSTM ensemble).
- **Advanced architectures:** Experiment with Transformers or NeuralProphet for long-horizon forecasting.

---

## **Conclusion**
This project successfully achieved the **Month 3 mini project objectives**, demonstrating:
- A solid pipeline for time-series forecasting.
- Integration of ML and deep learning models.
- A clear model comparison and insights on forecasting accuracy.

The results confirm that **LSTM is the most robust model** for capturing the temporal dynamics of electricity prices.

