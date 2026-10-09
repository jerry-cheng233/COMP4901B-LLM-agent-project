"""
Test script for UnifiedAgent - tests 5+ step trajectories using 3+ tools.
Demonstrates the use case from Part II Task 2 of the project.
"""
import sys
import json
sys.path.insert(0, 'src')
from unified_agent import UnifiedAgent

# API Keys
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

print("=" * 70)
print("Testing UnifiedAgent - 5+ Step Trajectories with 3+ Tools")
print("=" * 70)
print(f"\nTotal tools available: {len(agent.tools)}")
print("Tools: Notion (14) + Zoom (2) + Email (1) + Google Maps (4)")

# ============================================================================
# TEST: Multi-tool workflow - Notion + Google Maps + Email
# This demonstrates a 5+ step trajectory using 3+ different tools
# ============================================================================
print("\n" + "=" * 70)
print("TEST: Get event from Notion, find location on Google Maps, send email")
print("This should be 5+ steps using Notion, Google Maps, and Email tools")
print("=" * 70)

# Complex query that requires multiple tools
query = """
I have a comp4901 project meeting coming up.
1. First, check my Notion for any tasks related to 'comp4901 proj'
2. The meeting will be at HKUST. Search for HKUST on Google Maps and get directions from Kowloon Tong MTR
3. Also find a restaurant near HKUST for lunch after the meeting
4. Send an email to zihaochenglol@gmail.com with:
   - The task details from Notion
   - Directions to HKUST with Google Maps link
   - Restaurant recommendation near HKUST
"""

print(f"\nQuery: {query}")
print("\nProcessing...\n")

result = agent.run(query)

# Print full trajectory
print("=" * 70)
print("TRAJECTORY (Tool Calls)")
print("=" * 70)
for i, step in enumerate(result["steps"], 1):
    print(f"\nStep {i}: {step['tool']}")
    print(f"  Arguments: {json.dumps(step['arguments'], indent=4)[:200]}...")
    print(f"  Result preview: {step['result_preview'][:150]}...")

print("\n" + "=" * 70)
print("FINAL RESPONSE")
print("=" * 70)
print(result["final_response"][:1500])

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
tools_used = [s['tool'] for s in result['steps']]
unique_tools = list(set(tools_used))
print(f"Total steps: {len(result['steps'])}")
print(f"Tools used: {tools_used}")
print(f"Unique tools: {unique_tools} ({len(unique_tools)} different tools)")

# Check requirements
print("\n" + "-" * 70)
print("REQUIREMENTS CHECK:")
print(f"  ✅ 5+ steps: {len(result['steps'])} steps" if len(result['steps']) >= 5 else f"  ❌ Need 5+ steps, got {len(result['steps'])}")
print(f"  ✅ 3+ tools: {len(unique_tools)} tools" if len(unique_tools) >= 3 else f"  ❌ Need 3+ tools, got {len(unique_tools)}")

# Save trajectory to file for report
trajectory_output = {
    "query": query,
    "steps": result["steps"],
    "final_response": result["final_response"],
    "tools_used": tools_used,
    "unique_tools": unique_tools,
    "step_count": len(result["steps"]),
    "tool_count": len(unique_tools)
}

with open("results/trajectory_googlemaps_demo.json", "w") as f:
    json.dump(trajectory_output, f, indent=2, default=str)

print(f"\nTrajectory saved to: results/trajectory_googlemaps_demo.json")

