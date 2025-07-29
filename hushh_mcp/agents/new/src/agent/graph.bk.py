from ollama import create
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import Tool
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, create_react_agent
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor
import json
from typing import Dict, List, Callable

load_dotenv()

config = 'free'
llm_coding = create_react_agent("google_genai:gemini-2.0-flash" if config == 'free' else "anthropic:claude-4-sonnet-latest")
llm_planning = create_react_agent("google_genai:gemini-2.0-flash" if config == 'free' else "openai:gpt-4.1")


from langchain_google_community import GoogleSearchAPIWrapper
search = GoogleSearchAPIWrapper()


google_search = Tool(
    name="google_search",
    description="Use this tool to search the web for information. Provide a query and it will return relevant search results.",
    func=search.run,
)
tools = [google_search]


supervisor = create_supervisor(
    model=llm_planning.bind_tools(tools),
    agents=[],
    prompt=(
        "You are a supervisor. Your job is to break down tasks and assign them to agents.\n"
        "Use the `transfer_to_{agent}` tool to assign tasks.\n"
        "Available agents: `search_agent`, `coding_agent`.\n"
        "When done, return 'DONE'."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
    supervisor_name="supervisor"
)


def search_agent(state):
    system_msg = SystemMessage(content="You are a search agent. Perform searches and return summarized results.")
    messages = [system_msg] + state["messages"]
    result = llm_coding.invoke(messages)
    return {"messages": [result]}\
    

def make_coding_agent(state):
    system_msg = SystemMessage(content="You are a coding assistant. Write code to solve the user's problem.")
    messages = [system_msg] + state["messages"]
    result = llm_coding.invoke(messages)
    return {"messages": [result]}

class MyState(MessagesState):
    input: str

def supervisor_node(state: MyState):
    result = supervisor.invoke(state)
    last = result["messages"][-1].content.strip()
    state["last"] = last
    return {"input": last}

# Build graph
builder = StateGraph(MyState)


# Start ➜ Supervisor
builder.add_node("supervisor", supervisor_node)
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()

