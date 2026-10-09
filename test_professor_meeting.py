"""
Test script for UnifiedAgent - Professor Meeting & Deadline Extension Workflow.

This demonstrates a complex 7+ step trajectory using 4 different tools:
1. Notion - Read project notes and meeting summary
2. Google Maps - Find coffee shop near HKUST, get directions
3. Zoom - Create backup meeting
4. Email - Send comprehensive email to professor

Scenario: Student needs to request deadline extension due to project issues,
arrange a face-to-face meeting at a coffee shop, and provide backup Zoom link.
"""
import sys
import json
from datetime import datetime, timedelta

sys.path.insert(0, 'src')
from unified_agent import UnifiedAgent

# API Keys (same as test_unified_agent.py)
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"  # Replace with your actual DeepSeek API key
NOTION_API_KEY = "YOUR_NOTION_API_KEY_HERE"  # Replace with your actual Notion API key
ZOOM_CLIENT_ID = "YOUR_ZOOM_CLIENT_ID_HERE"  # Replace with your actual Zoom client ID
ZOOM_CLIENT_SECRET = "YOUR_ZOOM_CLIENT_SECRET_HERE"  # Replace with your actual Zoom client secret
ZOOM_ACCOUNT_ID = "YOUR_ZOOM_ACCOUNT_ID_HERE"  # Replace with your actual Zoom account ID
ZOOM_EMAIL = "YOUR_ZOOM_EMAIL_HERE"  # Replace with your actual Zoom email
GMAIL_EMAIL = "YOUR_GMAIL_EMAIL_HERE"  # Replace with your actual Gmail email
GMAIL_APP_PASSWORD = "YOUR_GMAIL_APP_PASSWORD_HERE"  # Replace with your actual Gmail app password
GOOGLE_MAP_API_KEY = "YOUR_GOOGLE_MAP_API_KEY_HERE"  # Replace with your actual Google Maps API key

# Initialize agent with all 4 tools
agent = UnifiedAgent(
    deepseek_api_key=DEEPSEEK_API_KEY,
    notion_api_key=NOTION_API_KEY,
    zoom_client_id=ZOOM_CLIENT_ID,
    zoom_client_secret=ZOOM_CLIENT_SECRET,
    zoom_account_id=ZOOM_ACCOUNT_ID,
    zoom_email=ZOOM_EMAIL,
    gmail_email=GMAIL_EMAIL,
    gmail_app_password=GMAIL_APP_PASSWORD,
    google_maps_api_key=GOOGLE_MAP_API_KEY
)

print("=" * 80)
print("UnifiedAgent Test: Professor Meeting & Deadline Extension Workflow")
print("=" * 80)
print(f"\nTotal tools available: {len(agent.tools)}")
print("Tools: Notion (14) + Zoom (2) + Email (1) + Google Maps (4)")
print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# ENHANCED QUERY: Complex multi-tool workflow requiring 7+ steps
# Tools needed: Notion (2+), Google Maps (2+), Zoom (1), Email (1)
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO: Request Deadline Extension & Schedule Professor Meeting")
print("Expected: 7+ steps using Notion, Google Maps, Zoom, and Email tools")
print("=" * 80)

query = f"""
To complete work in COMP 4901 part 3 notes in notion, we held a meeting, current issues are in Meeting summary in notion, send a email to professor Cheng on zihaochenglol@gmail.com explaining the issue, apply for deadline extension. 
Schedule a face to face discussion with professor, find a coffee shop near HKUST on Google Map, get direction to the coffee shop from HKUST. Also create a backup Zoom meeting in case of emergency. The email should include the deadline extension application information, directions to the coffee shop with Google Map link, backup Zoom link. Finally add the discussion meeting event back to notion todo list on the date you scheduled.
"""

print(f"\nQuery:\n{query}")
print("\n" + "-" * 80)
print("Processing... (this may take 30-60 seconds)")
print("-" * 80 + "\n")

# Run the agent
result = agent.run(query)

# Print detailed trajectory
print("=" * 80)
print("TRAJECTORY - Detailed Tool Calls")
print("=" * 80)

for i, step in enumerate(result["steps"], 1):
    print(f"\n{'─' * 40}")
    print(f"STEP {i}: {step['tool']}")
    print(f"{'─' * 40}")
    
    # Pretty print arguments
    args_str = json.dumps(step['arguments'], indent=2, ensure_ascii=False)
    if len(args_str) > 300:
        args_str = args_str[:300] + "\n  ... (truncated)"
    print(f"Arguments:\n{args_str}")
    
    # Show result preview
    result_preview = step['result_preview']
    if len(result_preview) > 200:
        result_preview = result_preview[:200] + "... (truncated)"
    print(f"\nResult: {result_preview}")

# Print final response
print("\n" + "=" * 80)
print("FINAL RESPONSE FROM AGENT")
print("=" * 80)
print(result["final_response"][:2000])
if len(result["final_response"]) > 2000:
    print("\n... (response truncated)")

# Summary and requirements check
print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

tools_used = [s['tool'] for s in result['steps']]
unique_tools = list(set(tools_used))
tool_categories = {
    "Notion": [t for t in tools_used if t in ["search_notes", "get_page_content", "get_tasks_today", 
               "get_tasks_tomorrow", "get_pending_tasks", "add_task", "get_task_details"]],
    "Google Maps": [t for t in tools_used if t in ["search_location", "get_directions", 
                   "get_place_info", "generate_map_link"]],
    "Zoom": [t for t in tools_used if t in ["create_zoom_meeting", "get_zoom_meeting"]],
    "Email": [t for t in tools_used if t == "send_email"]
}

print(f"\nTotal steps executed: {len(result['steps'])}")
print(f"All tools used (in order): {tools_used}")
print(f"Unique tools: {unique_tools} ({len(unique_tools)} different)")

print("\nTool breakdown by category:")
for category, tools in tool_categories.items():
    status = "✅" if tools else "⚠️"
    print(f"  {status} {category}: {len(tools)} calls - {tools if tools else 'Not used'}")

# Requirements check
print("\n" + "-" * 80)
print("PART II REQUIREMENTS CHECK:")
step_check = "✅" if len(result['steps']) >= 5 else "❌"
tool_check = "✅" if len(unique_tools) >= 3 else "❌"
print(f"  {step_check} 5+ steps requirement: {len(result['steps'])} steps")
print(f"  {tool_check} 3+ tools requirement: {len(unique_tools)} unique tools")

categories_used = sum(1 for tools in tool_categories.values() if tools)
cat_check = "✅" if categories_used >= 3 else "❌"
print(f"  {cat_check} Tool categories used: {categories_used}/4 (Notion, Maps, Zoom, Email)")

# Save trajectory
trajectory_output = {
    "test_name": "Professor Meeting & Deadline Extension Workflow",
    "timestamp": datetime.now().isoformat(),
    "query": query,
    "steps": result["steps"],
    "final_response": result["final_response"],
    "tools_used": tools_used,
    "unique_tools": unique_tools,
    "tool_categories": {k: len(v) for k, v in tool_categories.items()},
    "step_count": len(result["steps"]),
    "tool_count": len(unique_tools),
    "requirements_met": {
        "five_plus_steps": len(result['steps']) >= 5,
        "three_plus_tools": len(unique_tools) >= 3
    }
}

output_file = "results/trajectory_professor_meeting.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(trajectory_output, f, indent=2, default=str, ensure_ascii=False)

print(f"\n{'=' * 80}")
print(f"Trajectory saved to: {output_file}")
print(f"{'=' * 80}")

