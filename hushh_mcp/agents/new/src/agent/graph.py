import getpass
import os
from typing import TypedDict
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_supervisor import create_supervisor

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("GEMINI_API_KEY")

search = GoogleSearchAPIWrapper()
google_search = Tool(
    name="google_search",
    description="Use this tool to search the web for information. Provide a query and it will return relevant search results.",
    func=search.run,
)

planning_tools = [
    google_search,
]

tools = [
    google_search,
]


# TODO: Adding a planning agent, which is independant of langgraph
# TODO: This agent will be used to plan the execution of other agents

llm_planning = create_react_agent(
    model="google/gemini-1.5-flash",
    tools=planning_tools,
    prompt=(
        "You are a planning agent managing a variable number of agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
)

supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=tools,
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
) 

def create_coding_agents(name, llm: create_react_agent, tools: list[Tool]) -> create_react_agent:
    """Create coding agents with the given LLM and tools."""
    return create_react_agent(
        name=name,
        description="A coding agent that executes tasks assigned by the supervisor agent.",
        model=llm,
        tools=tools,
        prompt=(
            "You are a coding agent. Your task is to write code based on the instructions provided by the supervisor agent.\n"
            "You will receive tasks from the supervisor agent and you should execute them sequentially."
        ),
        name="coding_agent",
    )


class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    agent_results: list[dict]
    summary: str

class Int_State(TypedDict):
    input: str
    user_feedback: str
    messages: Annotated[list[str], add_messages]


## Without supervisor
graph_builder = StateGraph(
    Int_State,
    input_state=InputState,
    output_state=OutputState
)

## With supervisor
supervisor_graph_builder = create_supervisor(
    model=llm_planning,
    agents=[],
    prompt=(
        "You are a planning agent managing a variable number of agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
        "Make sure you delegate tasks in such a order that each agent gets only one function at a time."
        "Your aim is to create a plan that can be executed by the agents.\n"
        "You are a very good person and always try to help others, and reduce the load on individual agents\n"
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile

coding_agent = create_coding_agents(
    name="coding_agent",
    llm=llm_planning,
    tools=tools
)

planner = (
    StateGraph(
        Int_State,
        input_state=InputState,
        output_state=OutputState
    ).add_node(supervisor_agent, "planning_agent", destinations=("coding_agent", "search_agent"))
    .add_node(coding_agent)
    )