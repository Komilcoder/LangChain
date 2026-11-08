import json
from calendar import error

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
from langchain_core.messages import SystemMessage


@tool
def get_exchange_rate(currency: str)->str:
    """Return the exchange rate of the given currency (e.g. USD, EUR) to UZS."""
    rates = {"USD": "16000", "EUR": "14400"}
    return rates[currency.upper()]


@tool
def convert_amount(amount: float, currency: str)->str:
    """Convert the given amount to 100*12000=12000000 USD dollarni Uzbek sum please convert"""
    rates = {"USD": 12000, "EUR": 14400}
    total = amount * rates[currency.upper()]
    return f"{amount} {currency.upper()} = {total} UZS"


def main():
    question = input("savol:What is the USD exchange rate?")
    tools = {"convert_amount": convert_amount, "get_exchange_rate": get_exchange_rate}
    llm = ChatOllama(temperature=0, model="llama3.2:3b")
    llm_tools = llm.bind_tools(list(tools.values()))
    messages = [
    SystemMessage("You are an exchange rate assistant.\n"
    "Use the tools to get rates. Never use your own knowledge of rates.\n"
    "After a tool returns a result, reply with that result as a short sentence, "
    "keeping all numbers exactly as they are."),
    HumanMessage(question),
]

    for step in range(5):

        response = llm_tools.invoke(messages)
        # print(f"Step {step}: {response.tool_calls}")

        messages.append(response)

        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            tool = tools[tool_call["name"]]
            try:
                result = tool.invoke(tool_call['args'])
            except KeyError as e:
                result = f""
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        for m in messages:
            print(type(m).__name__, "->", m.content)
    # final=llm_tools.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()

