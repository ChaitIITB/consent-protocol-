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
llm_coding = init_chat_model("google_genai:gemini-2.0-flash" if config == 'free' else "anthropic:claude-4-sonnet-latest")
llm_planning = init_chat_model("google_genai:gemini-2.0-flash" if config == 'free' else "openai:gpt-4.1")


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

class DynamicState(MessagesState):
    input: Optional[str] = None
    agent_plan: Optional[Dict] = None
    dynamic_agents: Optional[Dict] = None

def create_dynamic_agent_function(agent_id: str, role: str, task: str) -> Callable:
    """Factory function to create agent functions"""
    def agent_function(state):
        system_msg = SystemMessage(content=f"""
You are {agent_id} with role: {role}
Your task: {task}
Complete your assigned task and provide results.
""")
        messages = [system_msg] + state["messages"]
        result = llm_coding.invoke(messages)
        return {"messages": [result]}
    
    return agent_function

def planning_supervisor(state: DynamicState):
    """Plans and creates the dynamic graph structure"""
    planning_prompt = """
Analyze the request and determine:
1. How many agents needed (2-10)
2. What role/task for each agent
3. Execution order

Respond in JSON:
{
    "agents": [
        {"id": "agent_1", "role": "Researcher", "task": "Research X"},
        {"id": "agent_2", "role": "Coder", "task": "Write code for Y"}
    ],
    "execution_order": ["agent_1", "agent_2"]
}
"""
    
    system_msg = SystemMessage(content=planning_prompt)
    messages = [system_msg] + state["messages"]
    result = llm_planning.invoke(messages)
    
    try:
        plan = json.loads(result.content)
        return {
            "messages": [result],
            "agent_plan": plan
        }
    except:
        # Fallback
        fallback_plan = {
            "agents": [
                {"id": "agent_1", "role": "Analyzer", "task": "Analyze the request"},
                {"id": "agent_2", "role": "Executor", "task": "Execute solution"}
            ],
            "execution_order": ["agent_1", "agent_2"]
        }
        return {
            "messages": [result],
            "agent_plan": fallback_plan
        }

def build_dynamic_graph(initial_state):
    """Build graph with dynamic nodes based on planning"""
    # First, run planning to get the structure
    planning_result = planning_supervisor(initial_state)
    agent_plan = planning_result.get("agent_plan", {})
    
    # Create a new graph builder
    dynamic_builder = StateGraph(DynamicState)
    
    # Add planning supervisor
    dynamic_builder.add_node("planning_supervisor", planning_supervisor)
    dynamic_builder.add_edge(START, "planning_supervisor")
    
    # Dynamically add agent nodes
    agents = agent_plan.get("agents", [])
    execution_order = agent_plan.get("execution_order", [])
    
    previous_node = "planning_supervisor"
    
    for i, agent_info in enumerate(agents):
        agent_id = agent_info["id"]
        role = agent_info["role"]
        task = agent_info["task"]
        
        # Create and add the agent node
        agent_function = create_dynamic_agent_function(agent_id, role, task)
        dynamic_builder.add_node(agent_id, agent_function)
        
        # Connect to previous node
        dynamic_builder.add_edge(previous_node, agent_id)
        previous_node = agent_id
    
    # Connect last agent to END
    if agents:
        dynamic_builder.add_edge(previous_node, END)
    else:
        dynamic_builder.add_edge("planning_supervisor", END)
    
    return dynamic_builder.compile()

# Usage function
def create_graph_for_request(user_input: str):
    """Create a custom graph for each user request"""
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    return build_dynamic_graph(initial_state)
