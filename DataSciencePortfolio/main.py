import streamlit as st
import asyncio
import plotly.io as pio
import ast
import altair as alt
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from call_function import download_client, download_pred_outcome, convert2png,get_heatmap, get_corr_filter, \
convert2base64, get_modellist, test_localmodel, updatemodel_env,\
updateAPIkey_env, get_openai_modellist, get_curr_model, get_curr_api;\
from model_file import MissingAPIKeyError
from llm_model import LLM_Model
import os

############################################################################
#initialize directory in environment
clientdatapath = os.getenv("client_storage")
serverdatapath = os.getenv("datasetfilepath")
plotfilepath = os.getenv("plot_filepath")

#initialize dataset
df = download_client(clientdatapath=clientdatapath,serverdatapath=serverdatapath)
y_pred = download_pred_outcome()

#filter features
feature_col1 = [col for col in df.columns if 'generation' in col]
feature_col2 = [col for col in df.columns if 'years' in col or 'months' in col]

#data splitting funciton
@st.cache_data()
def split(dataset):
    df_new=df.set_index('time')
    df_new.index = pd.to_datetime(df_new.index, utc=True)
    cutoff=pd.to_datetime('2018-01-01', utc=True)
    train = df_new[df_new.index < cutoff]
    test = df_new[df_new.index > cutoff]
    return train, test, df_new
@st.cache_data()
def b2S(bytedata):
    string_data = bytedata.decode(encoding='utf-8')
    nontype = [string_data] if string_data is not None else []
    if isinstance(nontype[0], str):
        nontype[0] = ast.literal_eval(nontype[0])
    return nontype
data = b2S(y_pred)
bf, af, full=split(df)
df_em_global=""

#edit sidebar width
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"]{
        width: 500px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
#streamlit ui title and config
async def stream_ui():
    st.set_page_config(layout='wide', page_title="Energy Price Forecast and Energy Analysis", initial_sidebar_state="collapsed")
    st.title("Energy Forecast Dashboard")
#streamlit sidebar
async def sidebar():
    global service_options
    global service_selection    
    with st.sidebar:
        st.header("API Configuration")
        service_options={"ollama":"ollama", "chatgpt":"OpenAI", "openrouter":"OpenRouter"}
        service_selection=st.selectbox(label="Services Selection", options=(service_options.keys()), key="provider_selection",  placeholder="Select your model")
        if "ollama" in service_selection :
            with st.spinner("Connecting to server...", show_time=True):
                localconnection=test_localmodel("http://127.0.0.1:11434")
                if localconnection["status_code"] == 200:
                    st.sidebar.success(localconnection["status_info"])
                else:
                    st.sidebar.error(localconnection['error_status'], localconnection['request_status_code'])                         
        else:
            if "chatgpt" in service_selection:
                options1= [get_curr_model(service_options[service_selection])['current_model']] + get_openai_modellist()
                modelselection = st.selectbox(label="LLM Model", options=(options1), key="model_selection", placeholder="choose your chatgpt model.")                 
                
            else:
                options1=[get_curr_model(service_options[service_selection])['current_model']] + get_modellist()['model_type']
                modelselection = st.selectbox(label="LLM Model", options=(options1), key="model_selection", placeholder="choose your chatgpt model.")
            updatemodel_env(modelselection, service_selection)#update model in environmental file whenever user selected the model 
            if not get_curr_api(service_options[service_selection])['current_api']:
                st.info(f"No, API records found.\n\nPlease Enter your {service_selection} API Key")          
            else:
                st.info("You already saved the api key, please proceed.")
            innerC = st.form("API_config",enter_to_submit=True,clear_on_submit=True)#create a form for user to update api key after click submit button
            api_input = innerC.text_input("API KEY",key='password', type='password', label_visibility='visible',help='Enter your LLM API key.')
            if innerC.form_submit_button("Confirm"):
                updateAPIkey_env(api_input, service_selection)#update apikey in environmental file whenever user enter the apikey  
                st.success("Your API key success fully saved in environment file.")      
#streamlit first content, include barchart, dataset and selectable variable,
async def content():            
    c1,c2 = st.columns([5,3])
    with c2.container(border=True, height=600):
        st.header("Feature selection")

        feature_selection1 = st.selectbox(
                "Feature_selection",
                (feature_col1)
            )   
        feature_selection2 = st.selectbox(
            "Feature selection (duration)", 
            (feature_col2)
        )
        
    with c1.container(border=True, height=600):
        st.header("Statistics and insights")
        with st.spinner(text="Loading data...", show_time=True):
            time.sleep(2)
        st.subheader(f"Generation Analysis: {feature_selection1} vs {feature_selection2}")
        #display bar chart
        st.bar_chart(df, x=feature_selection2, y=feature_selection1, color="years", use_container_width=True, height=500)  
        
    with c1.expander("Click to see insight..."):
        st.write("The Barchart shows the generated energy across the duration.")
        c3,c4 =st.columns([3,3])
        with c3.container(border=True,height=400):
            with st.spinner(text="Loading data...", show_time=True):
                time.sleep(5)
                grouped = df.groupby(feature_selection2)[feature_selection1].sum().reset_index()
                df_cus = pd.DataFrame({
                    f"generationtype: {feature_selection1}":grouped[feature_selection1],
                    "period": grouped[feature_selection2]
                    
                })
                st.table(df_cus) 
         
        with c4.container(border=True, height=100):
            df_metrics=pd.DataFrame({
                "Max value": [grouped[feature_selection1].values.max()],
                "Min value": [grouped[feature_selection1].values.min()],
                "Average value": [grouped[feature_selection1].values.mean()],
            })
            st.table(df_metrics)
    with st.container(border=True):
        st.header("Variable Selection for Correlation plot")
        categories_var = st.multiselect(label="variable",key="heatmap_var",options=get_heatmap().columns.tolist(), default=None)
        with st.container(height=200, border=False):
            c5,c6 = st.columns([2,2], border=True)
            with c5:   
                min_threshold = st.number_input(label="Minimum threshold:",key="heatmap_corr_minthreshold",min_value=0.0,max_value=0.9, step=0.1, value=0.4,format="%0.1f", width=200)
            with c6:
                max_threshold = st.number_input("Maximum threshold:", key="heatmap_corr_maxthreshold",min_value=0.0,max_value=0.9, step=0.1,value=0.9,format="%0.1f", width=200)
        if not categories_var:
            dfcorr=get_heatmap(minv=min_threshold, maxv=max_threshold)
        elif categories_var:
            dfcorr=get_heatmap(minv=min_threshold, maxv=max_threshold, variables=categories_var)

    with st.container(border=True):
        c7,c8=st.columns([3,2])   
        c7.header("Heatmap correlation")
        with c7:  
            fig=go.Figure( 
                data=go.Heatmap(
                z=dfcorr.values,
                x=dfcorr.columns.tolist(),
                y=dfcorr.columns.tolist()
            ))
            st.plotly_chart(fig, key='heatmap', use_container_width=True, theme='streamlit', config={'scrollZoom': False})
            convert2base64(fig, 'heatmap',plotfilepath)
        with c8:
            st.header("Correalation Results")
            options = [""]+get_corr_filter().columns.tolist()
            dfcorr_var=st.selectbox(label='Variable', options=options, index=0, format_func=lambda x: "Select Your correlation target variable." if x == "" else x, placeholder="Select Your correlation target variable.")
            if not dfcorr_var:
                dfcorr_table = get_corr_filter(minv=min_threshold, maxv=max_threshold)
            else:
                dfcorr_table = get_corr_filter(minv=min_threshold, maxv=max_threshold, variable=dfcorr_var)
            
            st.dataframe(dfcorr_table, use_container_width=True)

#streamlit second content, include price trend forecast, Evaluation metrics and AI insight on forecast model
async def graph():    
    with st.container(border=True, height=700): 
        st.subheader("Forecast price chart")
        #fig = px.line(df, x='time', y=df['price actual'], title='Time Series price 2015-2018')
        fig=go.Figure()
        fig.add_trace(go.Line(
            x=full.index,
            y=df['price actual'],
            name= 'Actual Price',
            yaxis= 'y1',
            line= dict(color='blue')
        ))
        fig.add_trace(go.Line(
            x=af.index,
            y=af['price actual'],
            name= 'Prediction Price',
            yaxis= 'y2',
            line= dict(color='magenta')
        ))
        fig.update_layout(
            title="Actual vs Forecast Price",
            xaxis=dict(title='Time'),
            yaxis=dict(title='Actual Price', side='left'),
            yaxis2= dict(title='Forecast Price', overlaying='y', side='right'),
            legend=dict(x=0.01,y=0.99),
            height=500
        )
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label="YTD", step='year', stepmode="todate"),
                    dict(count=1, label= "1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ))
        st.plotly_chart(fig, key="plot1", use_container_width=True, theme="streamlit")
   
    with st.expander("Click to see forecast price insight..."):
        st.write("The figure plot shows the forecasted energy price across the 5 years duration.")
        with st.container(border=True, height=300):
            st.header("Evaluation Metrics from XGBoost")
            df_em=pd.DataFrame({
                "RMSE": [float(data[0]["EM"][0])],
                "MAE": [float(data[0]["EM"][1])],
                "R2_score": [float(data[0]["EM"][2])]
            })
            df_em_global=df_em
            st.table(df_em)
        with st.container(border=True, height= 700):
            st.header("Insight summary")
            message=st.chat_message("assistant")
            message.write("Hello User!")
            message_placeholder=st.empty()
            message_placeholder.empty()
            convert2base64(fig,"forecast",plotfilepath)#send the figure to backend for png conversion
            with st.spinner(text="Loading Insight from your AI...", show_time=True):
                #setup model
                try:
                    if not get_curr_api(service_options[service_selection])["current_api"]:
                        raise MissingAPIKeyError("No API found in this service")
                    if st.button("process insight", key="Model"):
                        model_runner = LLM_Model(
                            input="Please provide me insight from the plot tell me what is the different between XGBoost prediction trend and price actual",
                            model_type= "openrouter",
                            template_path= os.getenv("template_filepath"),
                            image_path = os.getenv("img_filepath"),
                            model=os.environ['OpenRouter_model'],
                            Evaluation_metrics= df_em_global
                        )
                        response=model_runner.run()
                        message.write(response)
                except KeyError as e:
                    st.error(f"Configuration Error: {e}")
                except MissingAPIKeyError as e:
                    st.error(str(e))

#main functions           
async def main():
    await asyncio.gather(
        stream_ui(),
        sidebar(),
        content(),
        graph()
    )

if __name__ in "__main__":
    asyncio.run(main())