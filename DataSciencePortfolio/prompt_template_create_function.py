from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
import json



def build_prompt(template: str) -> str:
    """
    Replace placeholders in a multi-line prompt template and
    convert it into a single-line string with `\n` escapes.

    Args:
        template (str): Multi-line template containing placeholders like {var}
        **kwargs: Key-value pairs to replace placeholders

    Returns:
        str: Single-line prompt with placeholders replaced and newlines escaped
    """
    # Replace placeholders
    # Replace actual newlines with \n for JSON safety
    single_line_prompt = template.replace("\n", "\n")
    return single_line_prompt

def createlist(singlelinetemplate:str):
    """
    Convert str template into list
    Args:
        singlelinetemplate (str): the single line template as input.
    Returns:
        template format (list): return template in list format, [{"template_str": single line template}]
    """
    return [{"template_str": singlelinetemplate}]

###############################################################################################################
template =  """
You are a data scientist. Analyze the graph based on the following question.

--- Instructions ---
- Read the axis labels, legend, and data trends carefully.
- Identify key trends, peaks, or anomalies.
- Verify the evaluation metrics {Evaluation_metrics_key} of the prediction model.
- Provide a technical explanation based on the visual data.
- Do not speculate beyond the figure content.

---Analyze Condition---
- You must think first before provide technical explanation.
- You must understand the {Evaluation_metrics} provided in dataframe.
- You must verify the overall price trend based on the following question:
    1) How is the price trend movement over year.
    2) Does the price trend movement periodical
    3) Does the overall price trend increasing? or constantly fluctuating while remain the same amplitude?
    4) How does the price trend impact on energy consumption?
    5) How well is the forecast model performance compare to the actual price?
- Do not provide any speculate beyond the provided question.

--- Question ---
{question}

--- Output Format ---
Answer (in technical language, structured in bullet points or numbered steps):
"""
###############################################################################################################

#convert template into single line structure and insert necessary variable.
final_template=build_prompt(template=template)

#overwrite prompt template with new template
with open(r"C:\Users\PC\Desktop\program\DataSciencePortfolio\Server\Prompttemplate.json", "w") as f:
   json.dump(createlist(final_template),f)