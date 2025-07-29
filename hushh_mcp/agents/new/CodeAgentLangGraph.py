import dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display
from typing import Annotated
from langgraph.graph.message import add_messages
import mermaid
from langgraph.prebuilt import ToolNode

import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = "AIzaSyCPMRcMHRwf4TQc6DWAuZgQRu7cTMi5lJo"

config = 'free'

llm_coding = init_chat_model("google_genai:gemini-2.0-flash" if config == 'free' else "anthropic:claude-4-sonnet-latest")
# llm_planning = init_chat_model("openai:gpt-4.1")
llm_planning = init_chat_model("google_genai:gemini-2.0-flash" if config == 'free' else "openai:gpt-4.1")



class State(TypedDict):
    input: str
    user_feedback: str  
    messages: Annotated[list, add_messages]

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


@tool
def google_search(query: str) -> str:
    """Perform a Google search."""
    import requests
    URL = "https://www.googleapis.com/customsearch/v1"
    PARAMS = {"key": os.getenv("GOOGLE_SEARCH_API"), "q": query}
    r = requests.get(url = URL, params = PARAMS)
    data = r.json() 
    # This is a placeholder for the actual search logic
    # In practice, you would use a library or API to perform the search
    print(f"Searching Google for: {query}")
    return data.get("items", [{}])[0].get("snippet", "No results found.")

tools = [google_search]
llm_planning_with_tools = llm_planning.bind_tools(tools)

def planning_agent(state):
    message = llm_planning_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

def search_agent(state):
    message = llm_planning_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

def coding_agent(state):
    message = llm_coding.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

def human_feedback(state):
    print("---human_feedback---")
    feedback = interrupt("Please provide feedback:")
    return {"user_feedback": feedback}

tool_node = ToolNode(tools = tools)


builder = StateGraph(State)
builder.add_node("planning_agent", planning_agent)
builder.add_node("tools", tool_node)
builder.add_node("search_agent", search_agent)
builder.add_node("coding_agent", coding_agent)
builder.add_edge(START, "planning_agent")
builder.add_edge("planning_agent", "tools")
builder.add_edge("tools", "search_agent")
builder.add_edge("search_agent", "coding_agent")
builder.add_edge("coding_agent", END)

# Set up memory
memory = InMemorySaver()

# Add
graph = builder.compile(checkpointer=memory)

# View - Try without specifying draw method or use different approach
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not render graph: {e}")
    print("Graph structure:")
    print(graph.get_graph().draw_mermaid())

# user_input = "I need you to call the tool human assistance, can you do that for me?"

config = {"configurable": {"thread_id": "1"}}

# events = graph.stream(
#     {"messages": [{"role": "user", "content": user_input}]},
#     config,
#     stream_mode="values",
# )
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()



 
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config, stream_mode="values",):
        event["messages"][-1].pretty_print()
        snapshot = graph.get_state(config)
        print(snapshot)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break