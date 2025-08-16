from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
from typing import Optional
from openai import OpenAI
import openai
from PIL import Image
from IPython.display import HTML, display
from io import BytesIO
import base64
import requests
import os ,dotenv, json
from getpass import getpass
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
import uvicorn
#######################################################################
app=FastAPI()

#initialize API and model
#using local llm so no API if you have api please copy your api in .env
dotenv.load_dotenv(dotenv.find_dotenv()) #search local to verify .env
img_filepath = os.getenv("img_filepath") #get image_filepath
temp_path = os.getenv("template_filepath") #get template filepath 
if "OpenAI_API_KEY" not in os.environ or "OpenRouter_API_KEY" not in os.environ:
    os.environ["OpenAI_API_KEY"] = getpass
    os.environ["OpenRouter_API_KEY"] = getpass
#check does the required file path availabel
if os.path.exists(img_filepath) or os.path.exists(temp_path):
    print("created")
else:
    raise RuntimeError(f"No folder detect in {img_filepath} & {temp_path}")

#######################
#create image conversion to base64
def convert2base64(image:object):
    """
    Function to Convert PIL image to Base64 encoded strings
    
    Args:
        image (object): PIL image
    Returns:
        Resized base64 string
    """
    buffered =BytesIO()
    image.save(buffered, format="JPEG")
    img_str=base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

#display image in base64
def plt_img_base64(img_base64):
    """
    Function to display base64 encoded string as image
    Args:
        img_base64: converter image as base64
    Returns:
        image html (str): display image with base64 string
    """
    #create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}"/>'
    return display(HTML(image_html))

#check empty json
def safe_load_json(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

#OpenAI model list
@app.get("/openai/modellist")
def openaimodellist(openaimodellist:str=os.getenv("openaimodellist")):
    try:
        client=OpenAI()
        models = client.models.list()
        model_lists = [m.id for m in models]
        #update local json only if different
        lists = safe_load_json(openaimodellist)
        if model_lists != lists:
            with open(openaimodellist, 'w') as f:
                json.dump(model_lists,f, indent=2)
        return model_lists
    except:
        #Fallback to local JSON
        if os.path.exists(openaimodellist):
            with open(openaimodellist, 'r',encoding='utf-8') as f:
                return json.load(f)
        else:
            raise RuntimeError("No connection and no cached model list available.")

#extract prompt from template
def templatesetting(temp_path:str):
    with open(temp_path, "r", encoding="utf-8") as f:
        template=json.load(f)
    #setting llm analysis mode
    template = template[0]['template_str']
    return template
               
#language model pipeline                            
class LLM_Model():
    def __init__(self, input:str, model_type:str, template_path:str, image_path:str, Evaluation_metrics, model:Optional[str]=None):
        self.model_type = model_type
        self.template =self.load_template(template_path)
        self.input = input
        self.image_path= self.encode_image2base64(image_path)
        self.model = model
        self.Evaluation_metrics= self.convert_df2str(Evaluation_metrics)["Evaluation_metrics"]
        self.Evaluation_metrics_key = self.convert_df2str(Evaluation_metrics)["Evaluation_metrics_key"]
        dotenv.load_dotenv(dotenv.find_dotenv(),override=True)


    # encode image and convert to base64
    def encode_image2base64(self, image_path:str):
        with open(image_path, 'rb') as f:
            imagebase64 = json.load(f)
            return imagebase64['fig']

    #check template path and get template context
    def load_template(self, template_path:str):
        if template_path:
            with open(template_path,'r') as f:
                template_json = json.load(f)
                return template_json[0]['template_str']
            
    #convert dataframe to string
    def convert_df2str(self, Evaluation_metrics):
        Evaluation_metrics_key=Evaluation_metrics.keys()
        return {"Evaluation_metrics":Evaluation_metrics.to_string(index=False), "Evaluation_metrics_key":Evaluation_metrics_key}        
    # examine user selection on mode    
    def run(self):
        if self.model_type.lower() =='ollama':
            return self._run_ollama()
        elif self.model_type.lower() == 'openai':
            return self._run_openai()
        elif self.model_type.lower() == 'openrouter':
            return self._run_openrouter()
        else:
            raise ValueError("Unsupported model type. Use 'Ollama' or 'OpenAI'.")
    #run local llm
    def _run_ollama(self):
        '''
        Function calling vision language mode to analyse the figure diagram and provide insights.
        Returns:
            msg (str): LLM's Ouput after verifying the plot graph.
        '''
        try:
            prompt = ChatPromptTemplate.from_template(self.template)
            llm = OllamaLLM(model=os.getenv('LLM_Model'))
            if self.image_path:
                llm_with_image_context = llm.bind(images=[self.image_path])
            chain = prompt | llm_with_image_context
            msg=chain.invoke({"question":f"{input}", "Evaluation_metrics": f"{self.Evaluation_metrics}","Evaluation_metrics_key":f"{self.Evaluation_metrics_key}"})
            return msg
        except ConnectionError as e:
            assert {"status": 404, "Error Description": e}
    #run OpenAI
    def _run_openai(self):
        """
        Function to call openai through API.
        Returns:
            response (str): LLM's Ouput after verifying the plot graph.
        """
        openai.api_key=os.getenv("OpenAI_API_KEY")
        prompt = self.template.format(question=self.input, Evaluation_metric=self.Evaluation_metrics, Evaluation_metrics_key=self.Evaluation_metrics_key)
        image_base64=self.image_path
        try:
            response=openai.ChatCompletion.create(
                model=self.model_type,
                message=[
                    {
                        "role":"User",
                        "content":[
                            {"type":"text", "text": prompt},
                            {"type":"image_url", "image_url":
                            {"url":f"data:image/png;base64,{image_base64}"}}
                        ]
                    }
                ] 
            )
            return response["choice"][0]["message"]["content"]
        except ConnectionError as e:
            assert {"status": 404, "Status description": e }
    def _run_openrouter(self):
        """
        Function to call openrouter through API
        
        Returns:
            Response (str): LLM's output after verifying the plot graph
        """
        openrouter_api_key=os.environ["OpenRouter_API_KEY"]
        url ="https://openrouter.ai/api/v1/chat/completions"
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        prompt = self.template.format(question=self.input, Evaluation_metrics=self.Evaluation_metrics ,Evaluation_metrics_key=self.Evaluation_metrics_key)
        #Read and encode the image
        image_base64 = self.image_path
        data_url=f"data:image/png;base64,{image_base64}"
   
        messages=[
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": data_url   
            }
        ]
    }
]       
        try:
            payload = {
                "model": self.model,
                "messages": messages
            }
            
            res = requests.post(url,headers=headers,json=payload)
            return res.json()["choices"][0]["message"]["content"]
        except ConnectionError as e:
            assert {"status":res.status_code, "Error_description":e}
        
#run server at 0.0.0.0, port=8001
if __name__ in "__main__":
   uvicorn.run(app,host="0.0.0.0", port=8001)
