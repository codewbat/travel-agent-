import os
from dotenv import load_dotenv
import requests
from langchain_core.tools import tool , InjectedToolArg
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from typing import Annotated
load_dotenv()


EXCHANGE_API_KEY = os.getenv("Exchange_API_KEY")
groq_key = os.getenv("GROQAI_API_Key")


llm = ChatGroq(
    api_key=groq_key,
    model="llama-3.3-70b-versatile",
    temperature=0.2
)
@tool
def get_conversion_factor(
    base_currency : str,
    target_currency : str
) -> float:
    """
    this fuction is the currency coversion factor between a give base currency and target currency 
    """
    url = f'https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{base_currency}/{target_currency}'
    
    
    response = requests.get(url)
    
    data= response.json()
    return data

@tool
def convert(
    base_currency : float ,
    conversion_rate : Annotated[float, InjectedToolArg]
    # iska matlab ye hai ki ye dyminc hai jab tak iski value nhi jayegi jab tak converstion rate nhi milega api se 
) -> float :
    """
    given a currency conversion rate this function calculate  the target currency value from a give base currency value 
    """
    
    return base_currency * conversion_rate

# conversion_rate =get_conversion_factor.invoke({
#     'base_currency' : "INR",
#     'target_currency' : "USD"
# })

# converted_currency = convert.invoke({
#     'base_currency' : 25,
#     'conversion_rate' : conversion_rate
# })

# print(converted_currency)

# tool binding 

llm_with_tools = llm.bind_tools(
    [
        get_conversion_factor,
        convert
    ]
)

#tool calling 

message = input("Enter your prompt : \n")
chatconvo = []
chatconvo.append(HumanMessage(content=message))
ai_message = llm_with_tools.invoke(chatconvo)
chatconvo.append(ai_message)
# print(ai_message.tool_calls)
# chatconvo.append(tool_message)

import json

for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content) ['conversion_rate']
        chatconvo.append(tool_message1)
    if conversion_rate :
        if tool_call['name'] == 'convert':
            tool_call['args']['conversion_rate'] = conversion_rate
            tool_message2 = convert.invoke(tool_call)
            chatconvo.append(tool_message2)
    else:
        raise ValueError("Unexpeded error please try agian")
final_result = llm_with_tools.invoke(
    chatconvo
)
print(final_result.content)