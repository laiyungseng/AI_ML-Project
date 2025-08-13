import requests
import streamlit as st
import dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Optional

#Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv())
XGBMfilepath = os.getenv("XGBMfilepath")
datasetfilepath = os.getenv("datasetfilepath")
client_storage = os.getenv("client_storage")
modellistpath = r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\modellist.json"

#get file paths from environment variables
@st.cache_data()
def download_client():
    """
    Function to download dataset from server.

    Returns:
        df (pd.DataFrame): the contain of the dataset.
    """
    response=requests.get("http://127.0.0.1:8000/download_data/{energy_pd_clean.csv}",
                        params={"filename": "energy_pd_clean.csv"})
    if response.status_code == 200:
        with open(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\client-document\energy_pd_clean2.csv", "wb") as f:
            f.write(response.content)
        df = pd.read_csv(client_storage)
        df = pd.DataFrame(df)
        print(f"CSV download complete, responses status: {response.status_code}")
        return df
    else:
        return f"Download fail: {response.status_code}"

#get prediction value from XGB model in backend   
def download_pred_outcome():
    """
    Function to get prediction values from the Xgboost model in server

    Returns:
        y_pred (float): the forecast price from the xgb model
    """
    response = requests.get("http://127.0.0.1:8000/loadpred")
    if response.status_code == 200:
        y_pred= response.content
    else:
        return f"No prediction value, {response.status_code}"
    return y_pred

#png conversion function
def convert2png(fig:object):
    '''
    Function to send figure from frontend to backend through API calling.
    Convert to figure to json.

    Args:
        fig (object): figure plot that received as input
    '''
    fig_json = fig.to_json()
    response = requests.post("http://127.0.0.1:8000/llmconvert",json={"fig":fig_json})
    return 

#call api function to get model list.
def get_modellist():
    """
    Function to get model list from OpenRouter.

    Returns:
        model_detail (list): a dict of model details, {"model_type:[ ], "price_detail":[ ]}.
    """
    response = requests.get("http://127.0.0.1:8000/modellist/openrouter")
    openrouterlist=response.json()
    modellist = [m['model_id'] for m in openrouterlist]
    price_detail = [m['pricing_detail'] for m in openrouterlist]
    return {"model_type": modellist, "price_detail":price_detail}
print(get_modellist())
#send request to verify connection with local
def test_localmodel(ipconfig:str):
    """
    Function to test feedback from local server.

    Args:
        ipconfig (str): local ip address and port configurated in ollama or docker, (e.g."http://127.0.0.1:11434").
    Returns:
        Status (str): Successful connected status will shown if status code = 200.
    """
    try:
        ollama_req = requests.get(ipconfig)
        if ollama_req.status_code == 200:
            return {"status_info": "Successfully connected to local Server!","status_code": ollama_req.status_code}
    except ConnectionRefusedError or ConnectionError or RuntimeError as e:
        return {"error_status": e, "request_status_code":ollama_req.status_code }

#update env content based on the selection and info receive
def updatemodel_env(model:str, selected_service:str, dotenv_file:Optional[str]=None):
    """
    Function to call api for updating parameter in environment.

    Args:
        model(str): model of the LLM (e.g. chatgpt-3o, chatgpt-4o, chatgpt-4o-mini).
        selected_service(str): service provider user choose (e.g. ollama, chatgpt, openrouter)
        dotenv_file (str): directory of the .env (default=None).

    Returns:
        res (List): response reply from API, include status code and content.
    """
    res = requests.get("http://127.0.0.1:8000/env/updatemodel",
                   params={
                       "model":model,
                       "selected_service": selected_service,

                       "dotenv_file": dotenv_file
                   })
    return {"Request Status": res.status_code, "content": res.content}

#update env content based on the selection and info receive
def updateAPIkey_env(apikey:str, selected_service:str, dotenv_file:Optional[str]=None):
    """
    Function to call api for updating parameter in environment.

    Args:
        model(str): model of the LLM (e.g. chatgpt-3o, chatgpt-4o, chatgpt-4o-mini).
        selected_service(str): service provider user choose (e.g. ollama, chatgpt, openrouter)
        dotenv_file (str): directory of the .env (default=None).

    Returns:
        res (List): response reply from API, include status code and content.
    """
    res = requests.get("http://127.0.0.1:8000/env/updateAPIkey",
                   params={
                       "apikey":apikey,
                       "selected_service": selected_service,
                       "dotenv_file": dotenv_file
                   })
    return {"Request Status": res.status_code, "content": res.content}

#get openai function list
def get_openai_modellist():
    res=requests.get("http://127.0.0.1:8001/openai/modellist")
    return res.json()
