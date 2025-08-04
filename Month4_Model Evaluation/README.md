# 📊 Electricity Price Forecasting with Transformers, LSTM, and XGBoost
## 🔍 Project Overview
This project focuses on forecasting electricity prices using:

Traditional ML: XGBoost (with and without hyperparameter tuning)

Deep Learning: LSTM (Long Short-Term Memory)

Advanced Model: Temporal Fusion Transformer (TFT)

The primary objective is to:

Compare different forecasting approaches

Evaluate performance based on accuracy, computational cost, and scalability

Provide insights for real-world forecasting use cases (e.g., energy market predictions)

## 📂 Project Structure
```
Electricity_Forecasting_Transformer/
├── data/
│   └── electricity_data.csv
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_lstm_model.ipynb
│   ├── 3_transformer_model.ipynb
│   └── 4_model_comparison.ipynb
├── results/
│   ├── plots/
│   │   ├── actual_vs_predicted.png
│   │   ├── metrics_table.png
│   └── model_metrics.csv
├── src/
│   └── utils.py
├── README.md
└── requirements.txt
```
## 📦 Dataset
Source: Kaggle Electricity Load Forecasting Dataset

Features used:

Generation sources (coal, gas, renewables)

Total load forecast & actual load

Time-based features (hour, day, lag features)

Target variable:

Electricity Price (price actual)

## 🧠 Models Implemented
✔ Persistence (Baseline)

✔ XGBoost (Default)

✔ XGBoost (Optuna tuned)

✔ LSTM (Deep Learning)

✔ Temporal Fusion Transformer (Advanced)

## 📊 Model Evaluation Metrics

*Model Evaluation Metrics*
```
| Model                 | RMSE     | MAE      | R²       | Duration (s) |
|-----------------------|----------|----------|----------|--------------|
| Persistence           | 3.66068  | 13.40056 | 0.93358  | 0.01191      |
| XGBRegressor          | 2.60401  | 1.81935  | 0.94946  | 0.24644      |
| XGBRegressor (Optuna) | 2.56624  | 1.80295  | 0.94946  | 2.48322      |
| LSTM                  | 2.69008  | 1.88522  | 0.94606  | 84.00867     |
| LSTM (Optuna)         | 21.21086 | 19.97095 | -2.36108 | 678.86769    |
| TFT                   | 2.90478  | 2.43727  | 0.80552  | 1783.11225   |
```

## 📈 Visualization

*Actual vs Predicted (All Models)*

Model Evaluation Results

<img width="1009" height="448" alt="Image" src="https://github.com/user-attachments/assets/4f32e6b1-a1bb-44d8-84e7-fee9f92c235c" />

## ✅ Key Insights & Conclusion
Best Performer: Optimized XGBoost delivered the best RMSE (2.80) and shortest training time among advanced models.

Deep Learning Models: LSTM slightly outperformed TFT in accuracy, but both required higher computational resources.

Transformers’ Strength: TFT effectively captured temporal dependencies and is suitable for long-horizon forecasting.

Practical Implication: For real-time applications, XGBoost remains highly competitive due to speed and accuracy.

## ⚠ Limitations
Limited tuning for LSTM and TFT due to resource constraints

Weather and external factors not included

No live deployment pipeline yet

## 🔮 Future Improvements
Add weather data and holiday indicators for better accuracy

Explore Informer and N-BEATS models

Build Streamlit dashboard for visualization

Deploy model via FastAPI + Docker

## 🚀 How to Run
```
# Clone repository
git clone https://github.com/yourusername/Electricity_Forecasting_Transformer.git
cd Electricity_Forecasting_Transformer

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
```
