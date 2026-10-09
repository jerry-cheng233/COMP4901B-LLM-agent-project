#!/usr/bin/env python3
"""
Demo script for Part II: Zoom Agent - Task Management with Zoom Integration.

Module: run_zoom_agent.py

This script demonstrates 3 realistic tasks using the ZoomAgent:
1. Project Planning: Parse README tasks, split them, and schedule a kickoff meeting
2. Sprint Planning: Parse tasks, schedule them, and create milestone meetings
3. Task Review: Analyze tasks and schedule a review meeting

Each demonstration uses 3+ different tools and has 5+ steps.

Usage:
    python src/run_zoom_agent.py

Output:
    results/zoom_trajectories.json
"""

import os
import sys
import json
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zoom_agent import ZoomAgent

# Load credentials from key.txt if environment variables not set
def load_credentials():
    """Load API credentials from key.txt file."""
    key_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "key.txt")
    credentials = {}
    
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    credentials[key] = value
    
    return credentials


def print_trajectory(trajectory: dict, task_name: str):
    """Pretty print a trajectory for the report."""
    print(f"\n{'='*80}")
    print(f"TASK: {task_name}")
    print(f"{'='*80}")
    print(f"\nRequest: {trajectory['request']}")
    print(f"\n{'-'*40}")
    print("TRAJECTORY:")
    print(f"{'-'*40}")
    
    for step in trajectory["steps"]:
        step_num = step.get("step", "?")
        step_type = step.get("type", "unknown")
        
        if step_type == "thinking":
            print(f"\n[Step {step_num}] THINKING:")
            print(f"  {step['content'][:500]}..." if len(step.get('content', '')) > 500 else f"  {step.get('content', '')}")
        
        elif step_type == "tool_call":
            print(f"\n[Step {step_num}] TOOL CALL: {step['tool']}")
            print(f"  Arguments: {json.dumps(step['arguments'], indent=4, default=str)}")
            result = step.get('result', {})
            if isinstance(result, dict):
                if 'error' in result:
                    print(f"  Result: ERROR - {result['error']}")
                else:
                    # Truncate long results
                    result_str = json.dumps(result, indent=4, default=str)
                    if len(result_str) > 1000:
                        result_str = result_str[:1000] + "\n    ... (truncated)"
                    print(f"  Result: {result_str}")
            else:
                print(f"  Result: {result}")
        
        elif step_type == "final_response":
            print(f"\n[Step {step_num}] FINAL RESPONSE:")
            print(f"  {step['content']}")
    
    print(f"\n{'-'*40}")
    print(f"Total Steps: {trajectory.get('total_steps', len(trajectory['steps']))}")
    print(f"{'='*80}\n")


def demo_task_1(agent: ZoomAgent):
    """
    Demo Task 1: Project Planning Workflow
    
    Workflow: Parse README → Split first task → Schedule kickoff meeting
    Tools used: parse_tasks, split_task, create_meeting, get_meeting
    """
    request = """I need to plan our group project. Please:
1. First, parse the tasks from our README.md file
2. Take the first major task and split it into smaller subtasks
3. Create a Zoom kickoff meeting for tomorrow at 2pm to discuss the project plan
4. After creating the meeting, retrieve and confirm the meeting details"""
    
    print("\n" + "="*80)
    print("DEMO TASK 1: Project Planning Workflow")
    print("="*80)
    
    trajectory = agent.run(request)
    print_trajectory(trajectory, "Project Planning Workflow")
    return trajectory


def demo_task_2(agent: ZoomAgent):
    """
    Demo Task 2: Sprint Planning Workflow
    
    Workflow: Parse tasks → Schedule with deadline → Create sprint review meeting
    Tools used: parse_tasks, schedule_tasks, create_meeting, get_meeting
    """
    # Calculate deadline (project due date from README)
    deadline = "2025-12-04T23:59:00"
    meeting_time = (datetime.now() + timedelta(days=2)).replace(hour=15, minute=0).isoformat()
    
    request = f"""Help me with sprint planning:
1. Parse all tasks from README.md
2. Schedule these tasks with a deadline of {deadline}
3. Create a sprint review meeting on {meeting_time} to review progress
4. Get the meeting details to share with the team"""
    
    print("\n" + "="*80)
    print("DEMO TASK 2: Sprint Planning Workflow")
    print("="*80)
    
    trajectory = agent.run(request)
    print_trajectory(trajectory, "Sprint Planning Workflow")
    return trajectory


def demo_task_3(agent: ZoomAgent):
    """
    Demo Task 3: Task Breakdown and Meeting Workflow
    
    Workflow: Split complex task → Schedule subtasks → Create standup meeting
    Tools used: split_task, schedule_tasks, create_meeting, get_meeting
    """
    complex_task = "Implement a realistic agent with at least 3 tools for Part II of the project"
    deadline = "2025-12-04T20:00:00"
    meeting_time = (datetime.now() + timedelta(days=1)).replace(hour=10, minute=0).isoformat()
    
    request = f"""I have a complex task that needs breaking down:
Task: "{complex_task}"

Please:
1. Split this task into 5 smaller subtasks
2. Schedule these subtasks with deadline {deadline}
3. Create a daily standup meeting for {meeting_time} to track progress
4. Retrieve the meeting details for the team calendar"""
    
    print("\n" + "="*80)
    print("DEMO TASK 3: Task Breakdown and Meeting Workflow")
    print("="*80)

    trajectory = agent.run(request)
    print_trajectory(trajectory, "Task Breakdown and Meeting Workflow")
    return trajectory


def save_trajectories(trajectories: list, output_file: str):
    """Save all trajectories to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(trajectories, f, indent=2, default=str)
    print(f"\nTrajectories saved to {output_file}")


def main():
    """Run all demo tasks."""
    print("\n" + "#"*80)
    print("# Part II: Zoom Agent - Task Management with Zoom Integration")
    print("#"*80)

    # Load credentials
    creds = load_credentials()

    # Set environment variables if loaded from file
    if 'DEEPSEEK_API_KEY' in creds:
        os.environ['DEEPSEEK_API_KEY'] = creds['DEEPSEEK_API_KEY']
    if 'ZoomClientID' in creds:
        os.environ['ZOOM_CLIENT_ID'] = creds['ZoomClientID']
    if 'ZoomClientSecret' in creds:
        os.environ['ZOOM_CLIENT_SECRET'] = creds['ZoomClientSecret']
    if 'ZoomEmail' in creds:
        os.environ['ZOOM_EMAIL'] = creds['ZoomEmail']
    if 'ZoomAccountID' in creds:
        os.environ['ZOOM_ACCOUNT_ID'] = creds['ZoomAccountID']

    # For Server-to-Server OAuth, we need Account ID
    zoom_account_id = os.environ.get('ZOOM_ACCOUNT_ID', '')

    print("\nCredentials loaded:")
    print(f"  - DeepSeek API Key: {'✓' if os.environ.get('DEEPSEEK_API_KEY') else '✗'}")
    print(f"  - Zoom Client ID: {'✓' if os.environ.get('ZOOM_CLIENT_ID') else '✗'}")
    print(f"  - Zoom Client Secret: {'✓' if os.environ.get('ZOOM_CLIENT_SECRET') else '✗'}")
    print(f"  - Zoom Account ID: {'✓' if zoom_account_id else '✗ (Required for Zoom API)'}")
    print(f"  - Zoom Email: {os.environ.get('ZOOM_EMAIL', 'Not set')}")

    if not zoom_account_id:
        print("\n⚠️  WARNING: ZoomAccountID is not set!")
        print("   To use Zoom API, you need to:")
        print("   1. Go to https://marketplace.zoom.us/")
        print("   2. Create a Server-to-Server OAuth app")
        print("   3. Copy the Account ID from App Credentials")
        print("   4. Add it to key.txt as: ZoomAccountID=\"your-account-id\"")
        print("\n   The agent will still work, but Zoom meeting creation will fail.\n")

    # Initialize agent
    agent = ZoomAgent(
        api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
        zoom_client_id=os.environ.get('ZOOM_CLIENT_ID', ''),
        zoom_client_secret=os.environ.get('ZOOM_CLIENT_SECRET', ''),
        zoom_account_id=zoom_account_id,
        zoom_email=os.environ.get('ZOOM_EMAIL', '')
    )

    trajectories = []

    # Run Demo Task 1
    print("\n" + "~"*80)
    print("Running Demo Task 1: Project Planning Workflow...")
    print("~"*80)
    try:
        traj1 = demo_task_1(agent)
        trajectories.append({"task": "Project Planning", "trajectory": traj1})
    except Exception as e:
        print(f"Error in Demo Task 1: {e}")

    # Run Demo Task 2
    print("\n" + "~"*80)
    print("Running Demo Task 2: Sprint Planning Workflow...")
    print("~"*80)
    try:
        traj2 = demo_task_2(agent)
        trajectories.append({"task": "Sprint Planning", "trajectory": traj2})
    except Exception as e:
        print(f"Error in Demo Task 2: {e}")

    # Run Demo Task 3
    print("\n" + "~"*80)
    print("Running Demo Task 3: Task Breakdown and Meeting Workflow...")
    print("~"*80)
    try:
        traj3 = demo_task_3(agent)
        trajectories.append({"task": "Task Breakdown", "trajectory": traj3})
    except Exception as e:
        print(f"Error in Demo Task 3: {e}")

    # Save trajectories
    output_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results",
        "zoom_trajectories.json"
    )
    save_trajectories(trajectories, output_file)

    print("\n" + "#"*80)
    print("# Demo Complete!")
    print("#"*80)
    print(f"\nTotal trajectories: {len(trajectories)}")
    for i, t in enumerate(trajectories, 1):
        steps = t['trajectory'].get('total_steps', len(t['trajectory'].get('steps', [])))
        print(f"  {i}. {t['task']}: {steps} steps")


if __name__ == "__main__":
    main()
