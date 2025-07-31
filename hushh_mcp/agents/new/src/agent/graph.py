import getpass
import os
from typing import TypedDict, NotRequired
from langchain.chat_models import init_chat_model
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph_supervisor import create_supervisor
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from plannersup import plannergraph


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("GEMINI_API_KEY") 


from langgraph.types import Send


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool

assign_to_coding_agent_with_description = create_task_description_handoff_tool(
    agent_name="coding_agent",
    description="Assign task to a coding agent.",
)

assign_to_coding_agent2_with_description = create_task_description_handoff_tool(
    agent_name="math_agent",
    description="Assign task to 2nd coding agent.",
)



search = GoogleSearchAPIWrapper(google_api_key=os.environ.get("GOOGLE_API_KEY"),
                                google_cse_id=os.environ.get("GOOGLE_CSE_ID"))
google_search = Tool(
    name="google_search",
    description="Use this tool to search the web for information. Provide a query and it will return relevant search results.",
    func=search.run
    
)

planning_tools = [
    google_search,
    assign_to_coding_agent_with_description,
    assign_to_coding_agent2_with_description,
]

tools = [
    google_search,
]


# TODO: Adding a planning agent, which is independant of langgraph
# TODO: This agent will be used to plan the execution of other agents

llm_planning = init_chat_model(
    model="google_genai:gemini-2.0-flash",
    tools=planning_tools,
    prompt=(
        "You are a planning agent managing a variable number of agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
)

llm_coding = init_chat_model(
    model="google_genai:gemini-2.0-flash",
    tools=tools,
    prompt=(
        "You are a coding agent. Your task is to write code based on the instructions provided by the supervisor agent.\n"
        "You will receive tasks from the supervisor agent and you should execute them sequentially."
    ),
)

planning_agent = create_react_agent(
    model="google_genai:gemini-2.0-flash",
    tools= planning_tools,
    prompt=(
        "You are a supervisor managing two coding agents:\n"
        "You are to assign tasks to the coding agents based on the instructions provided.\n"
        "Call agents in parallel, and map tasks in such a way that each agent gets only one function at a time, and the tasks each agent recieves is sequential and independant from what other is doing\n"
        "Do not do any work yourself."
    ),
    name="planning_agent",
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
    )


class InputState(TypedDict):
    user_input: NotRequired[str] 
    messages: Annotated[list[str], add_messages]

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

coding_agent = create_coding_agents(
    name="coding_agent",
    llm=llm_coding,
    tools=tools
)

coding_agent2 = create_coding_agents(
    name="coding_agent2",
    llm=llm_coding,
    tools=tools
)

# With supervisor
graph = plannergraph(
    model=llm_planning,
    agents=[coding_agent, coding_agent2],
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
).compile()



# planner = (
#     StateGraph(
#         InputState,
#         input_state=InputState,
#         output_state=OutputState
#     ).add_node(planning_agent, destinations=("coding_agent", "coding_agent2", END))
#     .add_node(coding_agent, "coding_agent")
#     .add_node(coding_agent2, "coding_agent2")
#     .add_edge(START, "planning_agent")
#     .add_edge("coding_agent", "planning_agent")
#     .add_edge("coding_agent2", "planning_agent")
#     .compile()
#     )

# graph = planner
