#!/usr/bin/env python3
"""
Example usage of the dynamic agent system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.graph import run_dynamic_agents, get_agent_summary

def example_1_simple_coding_task():
    """Example 1: Simple coding task"""
    print("Example 1: Simple coding task")
    print("-" * 30)
    
    user_input = "Create a Python function to sort a list of dictionaries by a specific key"
    
    result = run_dynamic_agents(user_input)
    
    # Display results
    agent_results = result.get("agent_results", [])
    summary = get_agent_summary(agent_results)
    
    print(f"Number of agents executed: {summary['total_agents']}")
    print(f"Success rate: {summary['completion_rate']:.1%}")
    print(f"Agent roles: {', '.join(summary['agent_roles'])}")
    
    print("\nAgent outputs:")
    for i, agent_result in enumerate(agent_results, 1):
        print(f"\n{i}. {agent_result['agent_id']} ({agent_result['role']}):")
        print(f"   Task: {agent_result['task']}")
        print(f"   Result: {agent_result['result'][:200]}...")

def example_2_research_task():
    """Example 2: Research and analysis task"""
    print("\n\nExample 2: Research and analysis task")
    print("-" * 40)
    
    user_input = "Research the latest trends in artificial intelligence and create a summary report"
    
    result = run_dynamic_agents(user_input)
    
    # Display results
    agent_results = result.get("agent_results", [])
    summary = get_agent_summary(agent_results)
    
    print(f"Number of agents executed: {summary['total_agents']}")
    print(f"Success rate: {summary['completion_rate']:.1%}")
    print(f"Agent roles: {', '.join(summary['agent_roles'])}")

def example_3_complex_project():
    """Example 3: Complex multi-step project"""
    print("\n\nExample 3: Complex multi-step project")
    print("-" * 40)
    
    user_input = """
    Create a complete web scraping solution that:
    1. Scrapes news articles from a website
    2. Processes and cleans the text data
    3. Performs sentiment analysis
    4. Generates a visualization dashboard
    5. Creates documentation for the solution
    """
    
    result = run_dynamic_agents(user_input)
    
    # Display results
    agent_results = result.get("agent_results", [])
    summary = get_agent_summary(agent_results)
    
    print(f"Number of agents executed: {summary['total_agents']}")
    print(f"Success rate: {summary['completion_rate']:.1%}")
    print(f"Agent roles: {', '.join(summary['agent_roles'])}")
    
    # Show the execution flow
    print("\nExecution flow:")
    for i, agent_result in enumerate(agent_results, 1):
        status = "✓" if agent_result.get("success", True) else "✗"
        print(f"{i}. {status} {agent_result['agent_id']} ({agent_result['role']}): {agent_result['task']}")

if __name__ == "__main__":
    print("Dynamic Agent System - Usage Examples")
    print("=" * 50)
    
    try:
        example_1_simple_coding_task()
        example_2_research_task()
        example_3_complex_project()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your environment variables (.env file) with:")
        print("- GOOGLE_SEARCH_API")
        print("- GOOGLE_CSE_ID")
        print("- GOOGLE_API_KEY (for Gemini)")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
