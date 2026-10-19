from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

import os
import sys
from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    mcp_client = MultiServerMCPClient(
        {
            "maths": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable-http",
            }
        }
    )
    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
    tools = await mcp_client.get_tools()
    agent = create_agent(llm, tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 10 + 10?"}]}
    )
    print(result["messages"][-1].content)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is weather in delhi"}]}
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())