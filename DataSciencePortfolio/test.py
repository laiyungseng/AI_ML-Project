import os
import dotenv
import json

dotenv.load_dotenv()
plotpath = os.getenv("plot_filepath")

x=os.listdir(plotpath)


# "content": [
#             {
#                 "type": "text",
#                 "text": prompt
#             },
#             {
#                 "type": "image_url",
#                 "image_url": data_url   
#             }
#         ]
prompt="question"
outputs=[]
content=[{'type':"text","text": f"{prompt}"}]

for file in x:
    with open(plotpath+file, "r") as f:
        img64 = json.load(f)
    outputs.append({"type": "image_url", "image_url": {"url":f"data:image/png;base64,img64"}})
print(outputs)

content=content+outputs
print(content)
       