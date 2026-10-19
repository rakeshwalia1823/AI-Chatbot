"""Human-in-the-loop agent: pauses when human_assistance is called, then resumes."""

import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

load_dotenv()

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list[dict], add_messages]


@tool
def human_assistance(query: str) -> str:
    """Ask a human for help when you need clarification or approval or expert input.

    Args:
        query: clarification or approval or expert
    """
    # Pauses the graph here. Value is surfaced as result["__interrupt__"].
    # On resume, Command(resume=...) becomes the return value of interrupt().
    human_response = interrupt({"query": query})
    print(f"\n[Human assistance needed]\n{query}")
    return human_response["data"]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers and return their sum.

    Args:
        a: First number
        b: Second number
    """
    return a + b


llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilySearch(max_results=2)
tools = [tavily, add, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def tool_call_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("tool_call_llm", tool_call_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "tool_call_llm")
graph.add_conditional_edges("tool_call_llm", tools_condition)
graph.add_edge("tools", "tool_call_llm")  # after tools, go back to the LLM

graph.add_edge("tool_call_llm",END)

# checkpointer is required for interrupt / resume
app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "human-loop-1"}}

png = app.get_graph().draw_mermaid_png()
with open("humaninterupt_graph.png", "wb") as f:
    f.write(png)
print("Saved humaninterupt_graph.png")

def run_with_human_loop(user_message: str):
    result = app.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )

    # Keep asking the human until the graph finishes (no pending interrupts)
    while result.get("__interrupt__"):
        for item in result["__interrupt__"]:
            query = item.value.get("query", item.value)
            print(f"\n[Human assistance needed]\n{query}")

        human_reply = input("Your reply: ").strip()
        result = app.invoke(
            Command(resume={"data": human_reply}),
            config=config,
        )

    return result


if __name__ == "__main__":
    prompt = input("You: ").strip()
    result = run_with_human_loop(prompt)
    print("\n--- Final messages ---")
    for message in result["messages"]:
        message.pretty_print()
