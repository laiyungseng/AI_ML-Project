import requests
import streamlit as st
import dotenv
import os
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.io as pio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from typing import Optional
from io import BytesIO
import base64
#############################################################################################
#Load environment variables
dotenv.load_dotenv(dotenv.find_dotenv(), override=True)
XGBMfilepath1 = os.getenv("XGBMfilepath1")
XGBMfilepath2 = os.getenv("XGBMfilepath2")
datasetfilepath = os.getenv("datasetfilepath")#Server dataset file path
client_storage = os.getenv("client_storage")# Client dataset file path


#get file paths from environment variables
@st.cache_data()
def download_client(clientdatapath:str, serverdatapath:str):
    """
    Function to download dataset from server.

    Args:
        datapath1 (str): client datapath for storing dataset.
        datapath2 (str): Server datapath for storing dataset.

    Returns:
        df (pd.DataFrame): the contain of the dataset.
    """
    response=requests.get("http://127.0.0.1:8000/download_data/{energy_pd_clean.csv}",
                        params={"filename": "energy_pd_clean.csv", "serverdatapath": serverdatapath})
    if response.status_code == 200:
        with open(clientdatapath, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(clientdatapath)
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
    try:
        response = requests.get("http://127.0.0.1:8000/loadpred",
                                params={"serverfilepath":datasetfilepath,
                                        "XGBMfilepath1": XGBMfilepath1,
                                        "XGBMfilepath2": XGBMfilepath2})
        if response.status_code == 200:
            y_pred= response.content
            return y_pred
        elif response.status_code == 422:  
            raise KeyError(f"status:{response},  Wrong input variable, please check you request params.")
    except requests.exceptions.RequestException as e:
        assert {f"No prediction value, error {e}"}

#png conversion function
def convert2png(fig:object, img_name:str):
    '''
    Function to send figure from frontend to backend through API calling.
    Convert to figure to json.

    Args:
        fig (object): figure plot that received as input
    '''
    fig_json = fig.to_json()
    response = requests.post("http://127.0.0.1:8000/llmconvert",json={"fig":fig_json},params={'img_name': img_name})
    return 

#conversion function2 to base64
def convert2base64(fig:object, img_name:str, plotfilepath:str):
    """
    Convert figure to base64 format.

    Args:
        fig (object): figure from plot.
        img_name (str): image name for the plot.
    
    Returns:
        response (list): API response status from request post."""
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
    except AttributeError:
        pio.write_image(fig, buf, format="png", width=800, height=800)
    try:
        buf.seek(0)
        base64_fig=base64.b64encode(buf.read()).decode("utf-8")
        response = requests.post("http://127.0.0.1:8000/image/base64",json={"fig":base64_fig, "img_name": img_name}, params={"plotfilepath":plotfilepath})
        if response.status_code == 200:
            return {"status":response.json()['status'], "img_name":response.json()['image_name']}
        elif response.status_code == 422:
            return {"status": 422, "Error_descriptions": "Wrong input variable, please check function input variable"}
    except ConnectionError as e:
        assert {"status": response.status_code, "Error_description": e}

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
        print("ollama connection fail")
        assert {"error_status": e, "request_status_code":ollama_req.status_code }

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

#function to construct heatmap
def get_heatmap(minv:float=0.4, maxv:float=0.9,  df_path:str=os.getenv("datasetfilepath"), variables:Optional[list]=None):
    """
    create heatmap from the datasets.
    
    Args:
        minv (float): minimum threshold for filtering heatmap.
        maxv (float): maximum threshold for filtering heatmap.
        df_path (str): dataset file directory.
        variables [Optional] (str): variable from dataset that set for target variable.

    Returns:
        fig (object): return figure plot configurate of heatmap.    
    """
    #initialize dataset filepath
    df = pd.read_csv(df_path)
    df=df.reset_index().drop(['time','index'], axis=1)

    if variables !=None:
        variables=list(variables)
        dfcorr=df[variables].corr(numeric_only=True)
    else:
        dfcorr=df.corr(numeric_only=True)
    dfcorr_filter= dfcorr[(abs(dfcorr>minv)) & (abs(dfcorr<=maxv))]
    return dfcorr_filter

#function for filter correlation.
def get_corr_filter(minv:float=0.4, maxv:float=0.9, df_path:str=os.getenv("datasetfilepath"),variable:Optional[str]=None):
    """
    function to display filtered correlation of the variable from the dataset.
    
    Args:
        minv (float): minimum value of the threshold for correlation filteration.
        maxv (float): maximum value of the threshold for correlation filteration.
        df_path (str): dataset file directory.
        variable (str): [Optional] variable from dataset that set for target variable.
    
    Results:
        dfcorr_filter (pd.DataFrame): display the correlation based on the filteration setting.
    """

    df = pd.read_csv(df_path)
    df=df.reset_index().drop(['time','index'], axis=1)
    dfcorr=df.corr(numeric_only=True)    
    if variable is not None:
        dfcorr=df.corr(numeric_only=True)[variable]
    else:
        dfcorr=df.corr(numeric_only=True)
    dfcorr_filter= dfcorr[(abs(dfcorr>minv)) & (abs(dfcorr<=maxv))]
    return dfcorr_filter

#function to get current model in environment
def get_curr_model(model:str):
    """
    get current model through API request.

    Args:
        model (str): get service model from the environment file depends on the service selected.

    Returns:
        res (list): return requests status code and current model saved in environment.
    """
    try:
        res=requests.get("http://127.0.0.1:8000/getmodel",
                        params={
                            "model":model
                        })
        return res.json()
    except ConnectionError as e:
        return {"status": res.status_code, "Error_description": e}

#function to send api to get current api in environment
def get_curr_api(model:str):
    """
    Get current API Key through API requests.

    Args:
        model (str): selected model of the service. (e.g.: OpenAI or OpenRouter)
    
    Returns:
        requests (list): return requests result in json format
    """
    try:
        res=requests.get("http://127.0.0.1:8000/getapi",
                        params={
                            "model": model
                        })
        return res.json()
    except ConnectionError as e:
        return {"status":res.status_code, "Error Description": e}
