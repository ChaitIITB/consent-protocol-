#!/usr/bin/env python3
"""
Test script for the dynamic agent system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.graph import run_dynamic_agents, validate_agent_plan, get_agent_summary

def test_basic_functionality():
    """Test basic functionality of the dynamic agent system"""
    print("Testing dynamic agent system...")
    
    # Test case 1: Simple request
    test_input = "Create a simple Python function to calculate fibonacci numbers"
    
    try:
        result = run_dynamic_agents(test_input)
        print("✓ Basic execution completed successfully")
        
        # Check if we have agent results
        agent_results = result.get("agent_results", [])
        if agent_results:
            print(f"✓ {len(agent_results)} agents executed")
            
            # Get summary
            summary = get_agent_summary(agent_results)
            print(f"✓ Agent summary: {summary}")
        else:
            print("⚠ No agent results found")
            
    except Exception as e:
        print(f"✗ Error during execution: {e}")

def test_plan_validation():
    """Test agent plan validation"""
    print("\nTesting plan validation...")
    
    # Valid plan
    valid_plan = {
        "agents": [
            {"id": "agent_1", "role": "Researcher", "task": "Research topic"},
            {"id": "agent_2", "role": "Coder", "task": "Write code"}
        ],
        "execution_order": ["agent_1", "agent_2"]
    }
    
    # Invalid plan
    invalid_plan = {
        "agents": [{"id": "agent_1", "role": "Researcher", "task": "Research"}],
        "execution_order": ["agent_1", "agent_2"]  # agent_2 doesn't exist
    }
    
    assert validate_agent_plan(valid_plan) == True, "Valid plan should pass validation"
    assert validate_agent_plan(invalid_plan) == False, "Invalid plan should fail validation"
    
    print("✓ Plan validation tests passed")

if __name__ == "__main__":
    print("Dynamic Agent System Test Suite")
    print("=" * 40)
    
    test_plan_validation()
    test_basic_functionality()
    
    print("\n" + "=" * 40)
    print("Test suite completed!")
