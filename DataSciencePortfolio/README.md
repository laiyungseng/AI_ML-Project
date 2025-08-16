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
#if using linux or docker
#replace *C:\Users\PC\Desktop\program\DataSciencePortfolio => .
#filepath configuration
XGBMfilepath1 =C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\model\XGBModel.pkl
XGBMfilepath2 =C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\model\XGBModel.json
datasetfilepath= C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\document\energy_pd_clean.csv
client_storage = C:\Users\PC\Desktop\program\DataSciencePortfolio\client-document\energy_pd_clean2.csv
plot_filepath = C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\plot_figure\
img_filepath = C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\plot_figure\heatmap.json
template_filepath = C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\Prompttemplate.json
openaimodellist = C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\openaimodellist.json
openroutermodelist = C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\openrouterlist.json

#Local LLM configuration
LLM_model = "" #granite3.2-vision:2b
LLM_API_KEY = ""


#OpenAI configurati0on
OpenAI_model=''
OpenAI_API_KEY=''


#OpenRouter configuration
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

