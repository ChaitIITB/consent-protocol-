import getpass
import os
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import Tool

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

supervisor_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[assign_to_research_agent, assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)