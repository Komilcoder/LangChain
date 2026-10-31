import json
from pydantic import BaseModel, Field, ValidationError
import strip
from dotenv import load_dotenv
import os
import strip

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def get_exchange_rate(currency: str)->float:
    """Return the exchange rate of the given currency (e.g. USD, EUR) to UZS."""
    rates = {"usd": 12000, "eur": 14400}
    return rates[currency.lower()]

def main():

    llm = ChatOllama(temperature=0, model="llama3.2:3b")
    llm_tools = llm.bind_tools([get_exchange_rate])
    messages = [HumanMessage("What is the USD exchange rate?")]
    response = llm_tools.invoke(messages)
    messages.append(response)
    for tool_call in response.tool_calls:
        result = get_exchange_rate.invoke(tool_call['args'])
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    final=llm_tools.invoke(messages)
    print(final.content)

if __name__ == "__main__":
    main()

