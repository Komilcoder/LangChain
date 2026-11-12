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

RATES = {"USD": 12000, "EUR": 14400}

TRIPS = {
    "T-101": {"employee": "Ali Valiyev", "city": "Samarqand", "status": "approved",  "start": "2026-11-12"},
    "T-102": {"employee": "Ali Valiyev", "city": "Buxoro",    "status": "pending",   "start": "2026-11-20"},
    "T-103": {"employee": "Dilnoza Karimova", "city": "Toshkent", "status": "rejected", "start": "2026-11-15"},
}


@tool
def get_trip_status(request_id: str)->str:
    """return trip status for given trip id"""
    code = request_id.upper()
    trip = TRIPS[code]
    return f"Trip {code}: employee {trip['employee']}, city {trip['city']}, status {trip['status']}, starts {trip['start']}."


@tool
def get_employee_trips(employee_name: str)->str:
     """return all business trips of the given employee (e.g.'Ali Valiyev')."""
     found = []
     for code, trip in TRIPS.items():
         if trip["employee"].lower() == employee_name.lower():
             found.append(f"{code}: {trip['city']}, {trip['status']}, {trip['start']}")
     if not found:
         return f"Error: no trips found for '{employee_name}'"
     return f"Trips of {employee_name}: " + "; ".join(found)



def main():
    question = input("  savol:what is trip id?")
    tools = {"get_trip_status": get_trip_status, "get_employee_trips": get_employee_trips}
    llm = ChatOllama(temperature=0, model="llama3.2:3b")
    llm_tools = llm.bind_tools(list(tools.values()))
    messages = [
    SystemMessage("You are a business trip assistant.\n"
    "Use the tools to get trip information. Never use your own knowledge.\n"
    "After a tool returns a result, reply with that result as a short sentence, "
    "keeping all details exactly as they are.\n"
    "If there are several tool results, mention every one of them."),
    HumanMessage(question),
]
    used_tools = False
    for step in range(5):
        model = llm if used_tools else llm_tools
        response = model.invoke(messages)
        # print(f"Step {step}: {response.tool_calls}")

        messages.append(response)

        if not response.tool_calls:
            break
        for tool_call in response.tool_calls:
            print(tool_call, "======tooolll call=====")

            try:
                tool = tools[tool_call["name"]]
                result = tool.invoke(tool_call['args'])
            except Exception as e:

                result = f"Error: trip '{tool_call['args']}' was not found. Available trip ids: T-101, T-102, T-103."
            if not str(result).strip():
                result = "Error: the tool returned no result."
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        used_tools = True
        messages.append(HumanMessage("Using the tool result above, answer the user’s question in one short sentence"))
        for m in messages:
            print(type(m).__name__, "->", m.content)
    # final=llm_tools.invoke(messages)
    print(response.content)

if __name__ == "__main__":
    main()

