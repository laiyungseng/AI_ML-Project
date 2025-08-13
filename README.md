<<<<<<< HEAD
# Forecasting dashboard & LLM integration

## Summary
A comprehensive data science application that combines XGBoost-based time series prediction with LLM capabilities. Built with a modern tech stack featuring FastAPI and Streamlit, this application provides powerful data analysis tools, AI-driven insights, and interactive visualizations.

## Overview
The application implements a client-server architecture where:
- Backend handles ML model training, predictions, and LLM interactions
- Frontend provides an intuitive interface for data analysis and visualization
- Supports multiple LLM providers for AI-powered insights
- Features automated data processing and model training pipelines

## Key Features

### Machine Learning Capabilities
- Time series prediction using XGBoost
- Automated model training and evaluation
- Performance metrics tracking (RMSE, MAE, R2)
- Data preprocessing pipeline

### AI Integration
- Multi-provider LLM support
  - OpenAI
  - OpenRouter
  - Ollama
- Vision-language processing for graphs
- Context-aware analysis
- Custom prompt engineering

### Technical Features
- RESTful API architecture
- Real-time data visualization
- Secure file handling
- Environment-based configuration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DataSciencePortfolio.git
cd DataSciencePortfolio
```

2. Set up Python environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
copy .env.example .env
```

## Usage

1. Start the FastAPI server:
```bash
uv run python api_function.py
uv run python LLMmodel.py
```

2. Launch the Streamlit interface:
```bash
streamlit run main.py
```

## Project Structure

```
DataSciencePortfolio/
├── api_function.py      # API endpoints and ML functions
├── call_function.py     # Client-side function calls
├── LLMmodel.py         # LLM integration
├── main.py             # Application entry point
├── model_file.py       # ML model definitions
├── client-document/    # Client data storage
├── Server/             # Server resources
│   ├── document/      # Datasets
│   ├── model/        # Trained models
│   └── plot_figure/  # Visualizations
└── requirements.txt    # Project dependencies
```

## Technical Requirements

- Python 3.11+
- FastAPI
- Streamlit
- XGBoost
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Plotly
- OpenAI/OpenRouter/Ollama API access

## Configuration

Create a `.env` file with:
```env
MODEL_PATH=./Server/model
DATA_PATH=./Server/document
PLOT_PATH=./Server/plot_figure
#local
LLM_model = "granite3.2-vision:2b",
LLM_API_KEY = ""
#openai
OpenAI_model=''
OpenAI_API_KEY=''
#openrouter
OpenRouter_model=''
OpenRouter_API_KEY=''
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# AI_ML-Project
### AI/ML Learning Journey – Data Analysis, ML Models, and Time-Series Forecasting

## AI/ML Learning Portfolio
*Welcome to my AI/ML portfolio!*
This repository documents my journey of transitioning into the field of Artificial Intelligence and Machine Learning through a structured monthly learning plan.
Each month, I completed a mini project focusing on data science, machine learning, and deep learning concepts, including real-world forecasting challenges.

## Projects Overview
### *Month 1 – Data Analysis*
**Goal**: Learn dataset exploration, cleaning, and visualization.
Project: Electricity Consumption Data Analysis.

**Highlights:**

Data cleaning and preprocessing (handling missing values and outliers).

Exploratory Data Analysis (EDA) using Matplotlib and Seaborn.

Insights into energy consumption trends.

**Deliverables:**

Notebook: Electricity_consumption_Data_Analysis_tutorial.ipynb

Skills: pandas, matplotlib, seaborn

### **Month 2 – Machine Learning Models**
**Goal**: Implement machine learning models for forecasting tasks.
Project: Electricity Load Forecasting (Random Forest vs Baseline).

**Highlights:**

Feature selection based on correlation (generation, load, and price factors).

Model training with Random Forest and Linear Regression (baseline).

Added XGBoost for advanced performance comparison.

Metrics: RMSE, MAE, R².

Final comparison table to evaluate model performance.

**Deliverables:**

Notebook: DataPrediction_Tutorial2.ipynb

Skills: scikit-learn, xgboost, feature engineering, model evaluation

### **Month 3 – Time-Series Forecasting with Deep Learning**
**Goal**: Build time-series forecasting models using ML and DL techniques.
Project: Electricity Price Forecasting (LSTM vs XGBoost).

**Highlights:**

Created lag features (price_lag1, price_lag24) and rolling means.

Implemented XGBoost and Random Forest for time-series forecasting.

Developed LSTM (Long Short-Term Memory) model for sequence-based predictions.

Metrics comparison across Baseline, XGBoost, and LSTM.

Insights: LSTM outperformed other models in capturing temporal dependencies.

**Deliverables:**

Notebook: Tutorial_3_Deep_Learning.ipynb

Skills: time-series forecasting, tensorflow/keras, xgboost, sequence modeling

### **Month 4 – Advanced Forecasting with Transformers**
*Goal*: Explore advanced architectures like Temporal Fusion Transformer (TFT) and compare with traditional ML and LSTM models.
Project: Electricity Price Forecasting with Transformer Models.

**Highlights:**

Implemented Temporal Fusion Transformer (TFT) for time-series forecasting.

Compared Persistence, XGBoost (default), XGBoost (Optuna tuned), LSTM, and TFT.

Performed hyperparameter optimization using Optuna.

Visualized actual vs predicted prices for all models.

Metrics: RMSE, MAE, R², training time.

Key Insight: Optimized XGBoost achieved best accuracy with low computational cost; TFT offers scalability for multi-horizon forecasting.

**Deliverables:**

Notebook: Tutorial4_Model_comparison.ipynb

Skills: transformer models, pytorch forecasting, optuna, model interpretability

Sample Metrics Table:

Model	           |   RMSE	  |   MAE	   |   R²	   | Training Time (s)
-----------------|----------|----------|---------|-------------------|
Persistence      |	3.6606  |  13.400  |  0.933	 |      0.01191      |
XGBoost        	 |  2.6040	|  1.8193	 |  0.949  |      0.24644      |
XGBoost (Optuna) |	2.8000	|  1.8200	 |  0.949	 |      2.48322      | 
LSTM	           |  2.6900  |  1.8852  |  0.946	 |      84.00867     |
LSTM (Optuna     |  21.211  |  19.970  | -2.364  |      678.86769    |
TFT	             |  2.9047	|  2.437	 |  0.805	 |     1783.11225    |

**Skills Learned**  
✔ Data Analysis: Cleaning, visualization, EDA  
✔ Machine Learning: Feature engineering, Random Forest, XGBoost, hyperparameter tuning  
✔ Deep Learning: LSTM, sequence modeling, time-series forecasting  
✔ Advanced Models: Temporal Fusion Transformer (TFT), model interpretability  
✔ Model Evaluation: RMSE, MAE, R², MAPE, residual analysis  
✔ Tools & Libraries: Python, Pandas, Scikit-learn, TensorFlow/Keras, PyTorch Forecasting, Matplotlib, Seaborn, Optuna  

**Next Steps**
Explore Informer and N-BEATS models for further improvement.

Integrate weather and external factors into forecasting.

Develop a Streamlit dashboard for interactive visualization.

Deploy models via FastAPI + Docker for production-ready applications.

Repository Structure
```
AI_ML-Project/
├── Month1_Electricity_Analysis/
├── Month2_Load_Forecasting/
├── Month3_LSTM_Forecasting/
├── Month4_Transformer_Forecasting/
└── README.md
```
>>>>>>> 8b6ce05bb314dd386a96913fd0246e762da896d4
