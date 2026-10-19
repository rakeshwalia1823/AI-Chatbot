from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_groq import ChatGroq

import os 
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages: Annotated[list[dict], add_messages]

graph = StateGraph(State)

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

def chatbot(state: State):
    return {"messages": [ llm.invoke(state["messages"])]}

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

def prompt_user():
    user_input = input("Ask your question: ")
    return {"messages": [{"role": "user", "content": user_input}]}

app = graph.compile()

png = app.get_graph().draw_mermaid_png()
with open("chatbot_graph.png", "wb") as f:
    f.write(png)
print("Saved chatbot_graph.png")
result = app.invoke(prompt_user())
print(result["messages"][-1].content)