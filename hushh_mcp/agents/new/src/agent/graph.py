from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated, Optional, Dict, List, Callable
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, create_react_agent
import os
import json
import re
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

config = 'free'
llm_coding = init_chat_model("google_genai:gemini-2.0-flash" if config == 'free' else "anthropic:claude-4-sonnet-latest")
llm_planning = init_chat_model("google_genai:gemini-2.0-flash" if config == 'free' else "openai:gpt-4.1")

@tool
def google_search(query: str) -> str:
    """Search Google and return snippet."""
    import requests
    URL = "https://www.googleapis.com/customsearch/v1"
    PARAMS = {"key": os.getenv("GOOGLE_SEARCH_API"), "q": query, "cx": os.getenv("GOOGLE_CSE_ID")}
    r = requests.get(url=URL, params=PARAMS)
    data = r.json()
    return data.get("items", [{}])[0].get("snippet", "No results found.")

tools = [google_search]

class DynamicState(MessagesState):
    input: Optional[str] = None
    agent_plan: Optional[Dict] = None
    current_agent_index: int = 0
    agent_results: Optional[List[Dict]] = None

def planning_supervisor(state: DynamicState):
    """Plans and creates the dynamic agent structure"""
    planning_prompt = """
Analyze the request and determine:
1. How many agents needed (2-10)
2. What role/task for each agent
3. Execution order

Respond in JSON format:
{
    "agents": [
        {"id": "agent_1", "role": "Researcher", "task": "Research the topic thoroughly"},
        {"id": "agent_2", "role": "Coder", "task": "Write code based on research"},
        {"id": "agent_3", "role": "Reviewer", "task": "Review and improve the solution"}
    ],
    "execution_order": ["agent_1", "agent_2", "agent_3"]
}
"""
    
    system_msg = SystemMessage(content=planning_prompt)
    messages = [system_msg] + state["messages"]
    result = llm_planning.invoke(messages)
    
    try:
        # Clean the content to extract JSON
        content = result.content.strip()
        
        # Look for JSON block in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            plan = json.loads(json_str)
        else:
            # Try parsing the whole content
            plan = json.loads(content)
        
        # Validate the plan structure
        if not all(key in plan for key in ["agents", "execution_order"]):
            raise ValueError("Invalid plan structure")
        
        if not plan["agents"] or not plan["execution_order"]:
            raise ValueError("Empty agents or execution order")
        
        return {
            "messages": [result],
            "agent_plan": plan,
            "current_agent_index": 0,
            "agent_results": []
        }
    except Exception as e:
        print(f"Planning error: {e}. Using fallback plan.")
        # Fallback plan
        fallback_plan = {
            "agents": [
                {"id": "agent_1", "role": "Analyzer", "task": "Analyze the request thoroughly"},
                {"id": "agent_2", "role": "Executor", "task": "Execute the solution"}
            ],
            "execution_order": ["agent_1", "agent_2"]
        }
        return {
            "messages": [result],
            "agent_plan": fallback_plan,
            "current_agent_index": 0,
            "agent_results": []
        }

def dynamic_agent_executor(state: DynamicState):
    """Executes agents dynamically based on the plan"""
    agent_plan = state.get("agent_plan", {})
    current_index = state.get("current_agent_index", 0)
    agents = agent_plan.get("agents", [])
    execution_order = agent_plan.get("execution_order", [])
    agent_results = state.get("agent_results", [])
    
    # Validate inputs
    if not agents:
        print("Warning: No agents found in plan")
        return {
            "messages": state["messages"] + [SystemMessage(content="Error: No agents found in plan")],
            "current_agent_index": current_index
        }
    
    if not execution_order:
        print("Warning: No execution order found in plan")
        return {
            "messages": state["messages"] + [SystemMessage(content="Error: No execution order found in plan")],
            "current_agent_index": current_index
        }
    
    # Check if we've executed all agents
    if current_index >= len(execution_order):
        # All agents completed, compile final results
        final_summary = compile_agent_results(agent_results)
        return {
            "messages": state["messages"] + [SystemMessage(content=f"All agents completed. Final summary: {final_summary}")],
            "current_agent_index": current_index
        }
    
    # Get current agent to execute
    agent_id_to_execute = execution_order[current_index]
    current_agent = None
    
    for agent in agents:
        if agent["id"] == agent_id_to_execute:
            current_agent = agent
            break
    
    if not current_agent:
        # Skip if agent not found
        print(f"Warning: Agent {agent_id_to_execute} not found in agents list")
        return {
            "messages": state["messages"] + [SystemMessage(content=f"Warning: Agent {agent_id_to_execute} not found, skipping")],
            "current_agent_index": current_index + 1
        }
    
    # Create context from previous agent results
    context = create_agent_context(agent_results, current_agent)
    
    # Execute current agent with error handling
    try:
        system_msg = SystemMessage(content=f"""
You are {current_agent['id']} with role: {current_agent['role']}
Your task: {current_agent['task']}

Previous agent results:
{context}

Complete your assigned task and provide detailed results.
""")
        
        messages = [system_msg] + state["messages"]
        result = llm_coding.invoke(messages)
        
        # Store agent result
        agent_result = {
            "agent_id": current_agent['id'],
            "role": current_agent['role'],
            "task": current_agent['task'],
            "result": result.content,
            "index": current_index,
            "success": True
        }
        
    except Exception as e:
        print(f"Error executing agent {current_agent['id']}: {e}")
        # Store error result
        agent_result = {
            "agent_id": current_agent['id'],
            "role": current_agent['role'],
            "task": current_agent['task'],
            "result": f"Error executing agent: {str(e)}",
            "index": current_index,
            "success": False
        }
        result = SystemMessage(content=f"Error executing agent {current_agent['id']}: {str(e)}")
    
    updated_results = agent_results + [agent_result]
    
    return {
        "messages": state["messages"] + [result],
        "current_agent_index": current_index + 1,
        "agent_results": updated_results
    }

def create_agent_context(agent_results: List[Dict], current_agent: Dict) -> str:
    """Create context from previous agent results"""
    if not agent_results:
        return "No previous results available."
    
    context = "Previous agent outputs:\n"
    for result in agent_results:
        context += f"- {result['agent_id']} ({result['role']}): {result['result'][:200]}...\n"
    
    return context

def compile_agent_results(agent_results: List[Dict]) -> str:
    """Compile all agent results into a final summary"""
    if not agent_results:
        return "No agent results to compile."
    
    summary = "Final compilation of all agent results:\n\n"
    for result in agent_results:
        summary += f"**{result['agent_id']}** ({result['role']}):\n"
        summary += f"Task: {result['task']}\n"
        summary += f"Result: {result['result']}\n\n"
    
    return summary

def should_continue(state: DynamicState):
    """Determines if we should continue executing agents"""
    agent_plan = state.get("agent_plan", {})
    current_index = state.get("current_agent_index", 0)
    execution_order = agent_plan.get("execution_order", [])
    
    if current_index < len(execution_order):
        return "continue"
    else:
        return "end"

def build_dynamic_graph():
    """Build the dynamic graph structure"""
    # Create the graph builder
    builder = StateGraph(DynamicState)
    
    # Add nodes
    builder.add_node("planning_supervisor", planning_supervisor)
    builder.add_node("dynamic_agent_executor", dynamic_agent_executor)
    
    # Add edges
    builder.add_edge(START, "planning_supervisor")
    builder.add_edge("planning_supervisor", "dynamic_agent_executor")
    
    # Add conditional edge for looping through agents
    builder.add_conditional_edges(
        "dynamic_agent_executor",
        should_continue,
        {
            "continue": "dynamic_agent_executor",
            "end": END
        }
    )
    
    return builder.compile()

def create_graph_for_request(user_input: str):
    """Create a custom graph for each user request"""
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "agent_plan": None,
        "current_agent_index": 0,
        "agent_results": []
    }
    
    # Build and return the graph
    graph = build_dynamic_graph()
    return graph

# Usage example and additional utility functions
def run_dynamic_agents(user_input: str):
    """Run the dynamic agent system"""
    try:
        graph = create_graph_for_request(user_input)
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "agent_plan": None,
            "current_agent_index": 0,
            "agent_results": []
        }
        
        # Execute the graph
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        print(f"Error running dynamic agents: {e}")
        return {
            "messages": [SystemMessage(content=f"Error: {str(e)}")],
            "agent_plan": None,
            "current_agent_index": 0,
            "agent_results": []
        }

def validate_agent_plan(plan: Dict) -> bool:
    """Validate that an agent plan has the correct structure"""
    if not isinstance(plan, dict):
        return False
    
    if "agents" not in plan or "execution_order" not in plan:
        return False
    
    if not isinstance(plan["agents"], list) or not isinstance(plan["execution_order"], list):
        return False
    
    # Check that all agents in execution order exist in agents list
    agent_ids = {agent.get("id") for agent in plan["agents"] if isinstance(agent, dict) and "id" in agent}
    execution_ids = set(plan["execution_order"])
    
    return execution_ids.issubset(agent_ids)

def create_enhanced_agent_context(agent_results: List[Dict], current_agent: Dict) -> str:
    """Create enhanced context from previous agent results with role-specific information"""
    if not agent_results:
        return "No previous results available."
    
    current_role = current_agent.get("role", "").lower()
    context = f"Previous agent outputs (filtered for {current_role}):\n"
    
    for result in agent_results:
        if result.get("success", True):  # Only include successful results
            truncated_result = result["result"][:300] + "..." if len(result["result"]) > 300 else result["result"]
            context += f"- {result['agent_id']} ({result['role']}): {truncated_result}\n"
        else:
            context += f"- {result['agent_id']} ({result['role']}): [FAILED] {result['result']}\n"
    
    return context

# Additional utility functions for better agent coordination
def create_specialized_agent_function(agent_config: Dict) -> Callable:
    """Create specialized agent functions based on role"""
    role = agent_config.get("role", "").lower()
    
    def agent_function(state, task: str, context: str = ""):
        if "research" in role or "analyst" in role:
            system_msg = SystemMessage(content=f"""
You are a research specialist. Your task: {task}
Use available tools to gather information and provide comprehensive analysis.
Focus on factual accuracy and cite sources when possible.
Context: {context}
""")
        elif "code" in role or "develop" in role:
            system_msg = SystemMessage(content=f"""
You are a coding specialist. Your task: {task}
Write clean, efficient code with proper documentation.
Follow best practices and include error handling.
Context: {context}
""")
        elif "review" in role or "quality" in role:
            system_msg = SystemMessage(content=f"""
You are a quality assurance specialist. Your task: {task}
Review previous work and suggest improvements.
Focus on correctness, efficiency, and maintainability.
Context: {context}
""")
        else:
            system_msg = SystemMessage(content=f"""
You are {agent_config['id']} with role: {agent_config['role']}
Your task: {task}
Context: {context}
""")
        
        messages = [system_msg] + state["messages"]
        result = llm_coding.invoke(messages)
        return result
    
    return agent_function

def get_agent_summary(agent_results: List[Dict]) -> Dict:
    """Get a summary of agent execution results"""
    if not agent_results:
        return {"total_agents": 0, "successful": 0, "failed": 0, "completion_rate": 0.0}
    
    total = len(agent_results)
    successful = sum(1 for result in agent_results if result.get("success", True))
    failed = total - successful
    completion_rate = successful / total if total > 0 else 0.0
    
    return {
        "total_agents": total,
        "successful": successful,
        "failed": failed,
        "completion_rate": completion_rate,
        "agent_roles": [result.get("role", "Unknown") for result in agent_results]
    }