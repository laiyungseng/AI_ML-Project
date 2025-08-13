from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
import os, requests, uvicorn, dotenv,model_file,subprocess, asyncio, joblib, pickle, plotly.io.kaleido, json
from dotenv import load_dotenv, get_key, dotenv_values
from typing import Optional
import plotly.io as pio
from io import StringIO
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,r2_score

#################################################################################################################

#initialize FASTAPI app
app= FastAPI()

#Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())
#get file paths from environment variables
XGBMfilepath = os.getenv("XGBMfilepath")
datasetfilepath = os.getenv("datasetfilepath")

@app.get("/download_data/{filename}")
def download_data(filename:str):
    '''
    Function to download the dataset from local to API

    Args:
        filename (str): name of the dataset file
    Returns:
        None (str): return contain of the dataset.
    '''
    datasetfilepath = r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\document\energy_pd_clean.csv"

    if os.path.exists(datasetfilepath):
        return FileResponse(
            path=datasetfilepath,
            media_type='text/csv',
            filename=filename
        )
    else:
        return {"Error": "File not found"}
   
#load data
@app.post("/load_data")
async def uploaddata(file: UploadFile = File(...)):
    '''
    Function to load the dataset from a csv file
    Args:
        file (object): Path to the dataset csv file
    Returns:
        df (pd.DataFrame): Loaded DataFrame from the csv file
    '''
    await file.seek(0)
    contents = await file.read() 
    s = contents.decode('utf-8')
    data = StringIO(s)
    df = pd.DataFrame(data)
    return {'filename':file.filename,"rows":len(df)}

#load data
def load_data(datasetfilepath:str):
    '''
    Function to load the dataset from a csv file
    Args:
        datasetfilepath (str): Path to the dataset csv file
    Returns:
        df (pd.DataFrame): Loaded DataFrame from the csv file
    '''
    df = pd.read_csv(datasetfilepath)

    return df
 
#load mode
def load_model(XGBMfilepath:str):
    '''Function to load the XGBoost model from pkl
    Args:
        XGMBfilepath (str): Path to the XGBoost model file
    Returns:
        booster (xgboost.Booster): loaded XGBoost model
    '''
    #load XGBoost model
    model = joblib.load(XGBMfilepath)
    if xgb.__version__ <= '3.0.2':
        if not os.path.exists(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\model\XGBmodel.json"):
            print("No model.json found, converting model from old version to new version")
            try:
                # convert model to last version from xgboost, original version is 3.0.2
                reverted_lib = subprocess.run(['uv', 'remove', 'xgboost', '-y'],check=True, capture_output=True, text=True)
                convert_lib = subprocess.run(['uv','add', 'xgboost==3.0.2'], check=True, capture_output=True, text=True)
                booster = model.get_booster()
                booster.save_model(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\model\XGBmodel.json")
                if os.path.exists(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\model\XGBmodel.json"):
                    update_lib = subprocess.run(['uv', 'add', '--upgrade','xgboost'],check=True, capture_output=True, text=True)
            except (AttributeError, ImportError, Exception) as e:
                print(f"Error converting model: {e}")
        #load model from json
        booster = xgb.Booster()
        booster.load_model(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\model\XGBmodel.json")
    else:
        booster = model
    return booster

# data splitting and prediction function
def XGBoostpred(df, booster, test_size: float=0.2):
    '''
    Function to split data and predict with XGBoost model
    Args:
        df (pd.DataFrame): DataFrame containing the features and target variable  
        test_size (float): Proportion of the dataset to include in the test split  
        booster (xgboost.Booster): loaded XGBoost model
    Returns:
        Ttuple: A tuple containing the following elements:
           - y_pred (np.ndarray): Predicted values from the XGBoost model.
           - X_train (pd.DataFrame): Training features.
           - X_test (pd.Dataframe): Testing features.
           - y_train(pd.Series): Training target variable.
           - y_test(pd.Series): Testing target variable.
           - X (pd.DataFrame): All features used for prediction.
           - y (pd.Series): Target variable.
    '''
    fea_cols = [x for x in df.columns if 'price' in x or 'roll' in x or x in['hour', 'year']]
    X = df[fea_cols].drop(columns=['price day ahead', 'price actual'])
    y = df['price actual']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2)
    #dtest = xgb.DMatrix(X_test)#only suitable for xgboost 3.0.2

    #predict with model
    y_pred = booster.predict(X_test)
    return y_pred, X_train, X_test, y_train, y_test, X, y

# Plot graph to verify
def plot_pred_graph(X, X_train, y_train, y_test,y_pred):
    '''
    Function to plot the actual vs predicted values
    Args:
        X (pd.DataFrame): All features used for prediction
        X_train (pd.DataFrame): Training features
        y_train(pd.Series): Training target variable
        y_test(pd.Series): Testing target variable
        y_pred (np.ndarray): Predicted values from the XGBoost model
    Returns:
        None: Displays a plot of the actual vs predicted values
     
    '''
    plt.plot(X.index[:len(X_train)], y_train, label='Train data')
    plt.plot(X.index[len(X_train):], y_test, label='Test data', color='orange')
    plt.plot(X.index[len(X_train):], y_pred, label='Predicted data', color='red')

    plt.legend()
    plt.show()


# Evaluation Metrics (e.g., RMSE, MAE, R2_score)
def EvaluationMetrics(y_test, y_pred):
    """
    Function to calculate evaluation metrics

    Args:
        y_test (pd.Series): Testing target variable.
        y_pred (np.ndarray): Predicted values from the prediction model.

    Returns:
        tuple: A tuple contianing the evaluation metrics:
            -RMSE (float): Root Mean Squared Error.
            -MAE (float):  Mean Absolute Error.
            -R2_score (float): R-squared score.

    """
    XGB_RMSE = np.sqrt(root_mean_squared_error(y_test, y_pred))
    XGB_MAE = mean_absolute_error(y_test, y_pred)
    XGB_r2_score = r2_score(y_test, y_pred)

    # print("RMSE:",format(XGB_RMSE,'.3f'))
    # print("MAE:", format(XGB_MAE, '.3f'))
    # print("R2_score:", format(XGB_r2_score,'.3f'))    

    return XGB_RMSE, XGB_MAE, XGB_r2_score

#convertion
def safe_convert(obj):
    if hasattr(obj, 'values'):
        return obj.values.tolist()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj

#load dataset, mode, and make predictions with graphs and evaluation metrics
@app.get("/loadpred", response_model=model_file.PredictResponse)
def load_and_predict():
    '''
    Function to load the model and make predictions

    Args:
        df (pd.DataFrame): DataFrame containing the features and target variable
    Returns:
        None: Displays plot of actual vs predicted values and prints evaluation metrics.
    '''
    df = load_data(datasetfilepath)
    modelfilepath = XGBMfilepath
    model = load_model(XGBMfilepath=modelfilepath)
    var = XGBoostpred(df=df, booster=model)
    Evametrics=EvaluationMetrics(y_test=var[4], y_pred=var[0])
   
    output=model_file.PredictResponse(
        y_pred= safe_convert(var[0]),
        EM = safe_convert(Evametrics)
    )

    return output

#convert figure plot from json format to png
#save in local path folder "plot_figure"
@app.post("/llmconvert")
async def llmprocess(request:Request):
    '''
    Function to convert figure in Json format to PNG through Kalido library.
    Save the image to plot_figure folder in root directory
    Args:
        request (Request): Receive data from request.
    Returns:
        Status (str): status of succesffuly saved the image in JSON Format
    '''
    data = await request.json()
    fig_json = data['fig']
    fig = pio.from_json(fig_json)
    img=pio.write_image(fig, file=r'C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\plot_figure\figure.png', format='png' )
    return {"status": "figure saved"}

async def out_predict():
    df= load_data(datasetfilepath=datasetfilepath)
    model = load_model(XGBMfilepath=XGBMfilepath)
    var = XGBoostpred(df=df, booster=model)
    RMSE, MAE, R2_score = EvaluationMetrics(y_test=var[4], y_pred=var[0])
    return {'RMSE':RMSE, 'MAE':MAE, 'R2_score':R2_score}, var

@app.get("/modellist/openrouter")
#api calling to openrouter for retreiving all model lists.
def get_model_list():
    modellists=[]
    try:
        list_model = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={}, timeout=10
        )
        print(f"request status: {list_model.status_code}")
        with open(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\openrouterlist.json",'r') as f:
            model_details = json.load(f)
        if len(model_details['data']) < len(list_model.json()['data']):
            print("Update json file...")
            with open(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\openrouterlist.json", 'w') as f:
                json.dump(list_model.json(), f, indent=2)
            model_details=list_model.json()
        for item in range(len(model_details['data'])):
            modellist={
                    "model_id": model_details['data'][item]['id'],
                    "pricing_detail": model_details['data'][item]['pricing'],
                }
            modellists.append(modellist)
        
    except requests.exceptions.Timeout as e:
        print(f"TimeoutError: {e}")
        print("Retrieve from backup....")
        with open(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\openrouterlist.json",'r') as f:
            model_details=json.load(f)
        print(len(model_details['data']))
        for item in range(len(model_details['data'])):
            modellist={
                "model_id": model_details['data'][item]['id'],
                "pricing_detail": model_details['data'][item]['pricing'],
            }
            modellists.append(modellist)
       
        return modellists
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    return modellists

@app.get("/env/updatemodel")
#update model and apikey in environment file for chatgpt and openrouter
def environmentupdate(model:str,selected_service:str, dotenv_file:Optional[str]='.env'):
    """
    Function to update select model and apikey in environment file according to the User configuration.

    Args:
        model (str): Model version from the service provider. (e.g. chatgpt-3o-mini, chatgpt-4o, chatgpt-5)
        selected_service (str): user selected services. (e.g. ollama, chatgpt, openrouter)
        dotevn_file (Optional) (str): directory of the environment file.

    Returns:
        environ_update (List): echo the updated configuration in environment file
    """
   
    #search for environment file in the filepath
    dotenv_file=dotenv.find_dotenv()
    #read the content in the environment file
    config = dotenv_values(dotenv_file)
    
    #update the configuration parameter based on users input on selected services.
    if selected_service=='chatgpt':
        selected_service = 'OpenAI_model'
       
        if selected_service in config:
            dotenv.set_key(dotenv_file,selected_service,model)
            load_dotenv(dotenv_file,override=True)
            return {"OpenAI_model": os.environ[selected_service]}
        else:
            return 'no'
    elif selected_service == 'openrouter':
        selected_service = 'OpenRouter_model'
        if selected_service in config:
            dotenv.set_key(dotenv_file,selected_service,model)       
            load_dotenv(dotenv_file,override=True)
            return {"OpenRouter_model": os.environ[selected_service]}
        else:
            return 'No model file'

@app.get("/env/updateAPIkey")
def updateenvAPIkey(apikey:str, selected_service:str, dotenv_file:Optional[str]='.env'):
    """
    Function to update select model and apikey in environment file according to the User configuration.

    Args:
        selected_service (str): user selected services. (e.g. ollama, chatgpt, openrouter)
        apikey (str): the API key provided from service provider. (e.g. sasd-12lk-123l-daw1-sdml)
        dotevn_file (Optional) (str): directory of the environment file.

    Returns:
        environ_update (List): echo the updated configuration in environment file
    """
    
     #search for environment file in the filepath
    dotenv_file=dotenv.find_dotenv()
    #read the content in the environment file
    config = dotenv_values(dotenv_file)
        
    #update the configuration parameter based on users input on selected services.
    if selected_service=='chatgpt':
        default_api_key = "OpenAI_API_KEY"
        if default_api_key in config:
            if os.getenv('OpenAI_API_KEY') != "" or os.getenv("OpenAI_API_KEY") != apikey:
                dotenv.set_key(dotenv_file,default_api_key, apikey)
            load_dotenv(dotenv_file,override=True)
            return {"OpenAI_API_KEY": os.environ[default_api_key]}
        else:
            return 'no'
    elif selected_service == 'openrouter':
        default_api_key = "OpenRouter_API_KEY"
        if default_api_key in config:
            if os.getenv('OpenRouter_API_KEY') != "" or os.getenv("OpenRouter_API_KEY") != apikey:
                dotenv.set_key(dotenv_file,default_api_key, apikey)         
            load_dotenv(dotenv_file,override=True)
            return {"OpenRouter_API_KEY": os.environ[default_api_key]}
        else:
            return 'No model file'
##run test api
#fastapi dev C:\Users\PC\Desktop\program\DataSciencePortfolio\api.py
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port= 8000)