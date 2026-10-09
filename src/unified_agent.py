"""
Unified Multi-Tool Agent - Combines Notion, Zoom, Email, and Google Maps capabilities.

This agent integrates four separate tools:
1. Notion - Task management, checklist, notes
2. Zoom - Meeting scheduling
3. Email - Send emails via Gmail
4. Google Maps - Location search, directions, place information

Use Cases:
1. GET and POST from Notion to make checklist
2. Read info from Notion and schedule Zoom through email tool
3. Send email to professor about issues related to Notion information
4. Get events with locations from Notion, search on Google Maps, send directions via email
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from openai import OpenAI

# Import the individual tool classes
from notion_agent import Notion
from zoom_agent import (
    get_zoom_access_token, create_zoom_meeting, get_meeting_details,
    parse_tasks_from_markdown, split_task_into_subtasks, schedule_tasks_with_deadlines
)
from email_agent import send_gmail
from google_maps_agent import GoogleMapsClient, get_google_maps_tool_definitions

# Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class UnifiedAgent:
    """
    Unified Multi-Tool LLM Agent.

    Combines Notion, Zoom, Email, and Google Maps capabilities into a single agent
    that can handle complex multi-step tasks.
    """

    def __init__(
        self,
        deepseek_api_key: str = None,
        notion_api_key: str = None,
        zoom_client_id: str = None,
        zoom_client_secret: str = None,
        zoom_account_id: str = None,
        zoom_email: str = None,
        gmail_email: str = None,
        gmail_app_password: str = None,
        google_maps_api_key: str = None,
        model: str = "deepseek-chat",
        max_steps: int = 15,
        max_tokens: int = 2048
    ):
        # Load from environment if not provided
        self.deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.notion_api_key = notion_api_key or os.environ.get("NOTION_API_KEY", "")
        self.zoom_client_id = zoom_client_id or os.environ.get("ZOOM_CLIENT_ID", "")
        self.zoom_client_secret = zoom_client_secret or os.environ.get("ZOOM_CLIENT_SECRET", "")
        self.zoom_account_id = zoom_account_id or os.environ.get("ZOOM_ACCOUNT_ID", "")
        self.zoom_email = zoom_email or os.environ.get("ZOOM_EMAIL", "")
        self.gmail_email = gmail_email or os.environ.get("GMAIL_EMAIL", "")
        self.gmail_app_password = gmail_app_password or os.environ.get("GMAIL_APP_PASSWORD", "")
        self.google_maps_api_key = google_maps_api_key or os.environ.get("GOOGLE_MAP_API_KEY", "")

        # Initialize LLM client
        self.client = OpenAI(api_key=self.deepseek_api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        # Initialize Notion client
        self.notion = Notion(api_key=self.notion_api_key) if self.notion_api_key else None

        # Initialize Google Maps client
        self.google_maps = GoogleMapsClient(api_key=self.google_maps_api_key) if self.google_maps_api_key else None

        # Cache for Zoom token
        self._zoom_token = None

        # Build combined tool definitions
        self.tools = self._build_tool_definitions()
    
    def _get_zoom_token(self) -> str:
        """Get or refresh Zoom access token."""
        if not self._zoom_token:
            self._zoom_token = get_zoom_access_token(
                self.zoom_client_id,
                self.zoom_client_secret,
                self.zoom_account_id
            )
        return self._zoom_token
    
    def _build_tool_definitions(self) -> List[Dict]:
        """Build combined tool definitions from all agents."""
        tools = []
        
        # ===== NOTION TOOLS =====
        if self.notion:
            tools.extend(self.notion.get_tool_definitions())
        
        # ===== ZOOM TOOLS =====
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "create_zoom_meeting",
                    "description": "Create a Zoom meeting for scheduling a call/meeting",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Meeting topic/title"},
                            "start_time": {"type": "string", "description": "Start time in ISO 8601 format (e.g., 2024-12-05T10:00:00)"},
                            "duration": {"type": "integer", "description": "Duration in minutes (default: 60)"},
                            "agenda": {"type": "string", "description": "Meeting agenda/description"}
                        },
                        "required": ["topic", "start_time"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_zoom_meeting",
                    "description": "Get details of an existing Zoom meeting",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "meeting_id": {"type": "string", "description": "The Zoom meeting ID"}
                        },
                        "required": ["meeting_id"]
                    }
                }
            }
        ])
        
        # ===== EMAIL TOOLS =====
        tools.append({
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email via Gmail. Use this to notify people about meetings, tasks, or any information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string", "description": "Recipient email address(es), comma-separated for multiple"},
                        "subject": {"type": "string", "description": "Email subject line"},
                        "body": {"type": "string", "description": "Email body content"}
                    },
                    "required": ["recipient", "subject", "body"]
                }
            }
        })

        # ===== GOOGLE MAPS TOOLS =====
        if self.google_maps:
            tools.extend(get_google_maps_tool_definitions())

        return tools

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the unified agent."""
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""You are a powerful multi-tool assistant with access to Notion, Zoom, Email, and Google Maps.
Today's date is {today}.

## Your Capabilities:

### NOTION (Task Management & Notes)
- Query tasks from To-do Lists calendar (today, tomorrow, next week, date ranges)
- Search for notes and page content
- Add new tasks with due dates and priorities
- Update task status, priority, due dates
- Read full page content for details

### ZOOM (Meeting Scheduling)
- Create Zoom meetings with topic, time, duration, agenda
- Get meeting details and join links

### EMAIL (Communication)
- Send emails to anyone (professors, teammates, etc.)
- Include meeting links, task information, location directions, etc.

### GOOGLE MAPS (Location & Directions)
- Search for any place (restaurants, landmarks, addresses)
- Get detailed place info (ratings, hours, reviews, phone)
- Get directions between locations with step-by-step navigation
- Generate clickable Google Maps links for locations and directions

## Common Workflows:

1. **Schedule a meeting about a task**:
   - Search Notion for task details
   - Create a Zoom meeting
   - Send email with meeting link and agenda

2. **Email professor about a task**:
   - Get task details from Notion
   - Compose and send professional email

3. **Create checklist from meeting**:
   - Add tasks to Notion with due dates

4. **Send event location with directions**:
   - Get tasks/events from Notion that have locations
   - Search the location on Google Maps
   - Get directions from a starting point
   - Send email with event details and Google Maps link

5. **Find nearby places for an event**:
   - Get event details from Notion
   - Search for restaurants/venues near the location
   - Include recommendations in email

When dates are relative (like "tomorrow", "next Monday", "Dec 15"), convert to YYYY-MM-DD format.
When sending emails, be professional and include all relevant context including Google Maps links when locations are involved.
Always provide clear summaries of actions taken."""

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool and return its result."""

        # ===== NOTION TOOLS =====
        if self.notion and tool_name in [
            "get_tasks_today", "get_tasks_tomorrow", "get_tasks_next_week",
            "get_tasks_by_date_range", "get_pending_tasks", "get_overdue_tasks",
            "add_task", "mark_task_done", "get_task_details", "search_notes",
            "get_page_content", "add_note_to_task", "update_task_priority",
            "update_task_due_date"
        ]:
            return self.notion.execute_tool(tool_name, arguments)

        # ===== ZOOM TOOLS =====
        elif tool_name == "create_zoom_meeting":
            return create_zoom_meeting(
                topic=arguments.get("topic", "Meeting"),
                start_time=arguments.get("start_time"),
                duration=arguments.get("duration", 60),
                agenda=arguments.get("agenda", ""),
                access_token=self._get_zoom_token(),
                user_email=self.zoom_email
            )

        elif tool_name == "get_zoom_meeting":
            return get_meeting_details(
                arguments.get("meeting_id"),
                self._get_zoom_token()
            )

        # ===== EMAIL TOOLS =====
        elif tool_name == "send_email":
            return send_gmail(
                sender=self.gmail_email,
                password=self.gmail_app_password,
                recipient=arguments.get("recipient"),
                subject=arguments.get("subject"),
                body=arguments.get("body")
            )

        # ===== GOOGLE MAPS TOOLS =====
        elif tool_name == "search_location":
            return self.google_maps.search_places(
                query=arguments.get("query"),
                location=arguments.get("location")
            )

        elif tool_name == "get_place_info":
            return self.google_maps.get_place_details(
                place_id=arguments.get("place_id")
            )

        elif tool_name == "get_directions":
            return self.google_maps.get_directions(
                origin=arguments.get("origin"),
                destination=arguments.get("destination"),
                mode=arguments.get("mode", "driving")
            )

        elif tool_name == "generate_map_link":
            location = arguments.get("location")
            from_location = arguments.get("from_location")
            mode = arguments.get("mode", "driving")

            if from_location:
                link = self.google_maps.generate_directions_link(from_location, location, mode)
                return {"type": "directions", "link": link, "from": from_location, "to": location, "mode": mode}
            else:
                link = self.google_maps.generate_place_link(location)
                return {"type": "place", "link": link, "location": location}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def run(self, user_message: str) -> Dict[str, Any]:
        """
        Run the agent to handle a user request.

        Args:
            user_message: The user's natural language request

        Returns:
            Dictionary with response and trajectory
        """
        trajectory = {
            "user_message": user_message,
            "steps": [],
            "final_response": ""
        }

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_message}
        ]

        for step in range(self.max_steps):
            # Call LLM with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=self.max_tokens
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if we have tool calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Execute the tool
                    result = self._execute_tool(tool_name, arguments)

                    # Record in trajectory
                    trajectory["steps"].append({
                        "step": step + 1,
                        "tool": tool_name,
                        "arguments": arguments,
                        "result_preview": str(result)[:500]
                    })

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)
                    })
            else:
                # No tool calls - we have a final response
                trajectory["final_response"] = assistant_message.content
                break

        # If we exhausted steps without final response
        if not trajectory["final_response"]:
            trajectory["final_response"] = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

        return trajectory

    def chat(self, user_message: str) -> str:
        """Simple chat interface - returns just the response text."""
        result = self.run(user_message)
        return result["final_response"]


# ============================================================================
# Interactive Demo
# ============================================================================

if __name__ == "__main__":
    # Load API keys from key.txt
    import sys

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # key.txt is in the parent directory (project root)
    key_file_path = os.path.join(script_dir, "..", "key.txt")

    # Read keys
    keys = {}
    try:
        with open(key_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    keys[key.strip()] = value.strip().strip('"')
    except FileNotFoundError:
        print(f"key.txt not found at {key_file_path}. Please set environment variables.")
        sys.exit(1)

    # Initialize agent
    agent = UnifiedAgent(
        deepseek_api_key=keys.get("DEEPSEEK_API_KEY"),
        notion_api_key=keys.get("NOTION_API_KEY"),
        zoom_client_id=keys.get("ZoomClientID"),
        zoom_client_secret=keys.get("ZoomClientSecret"),
        zoom_account_id=keys.get("ZoomAccountID"),
        zoom_email=keys.get("ZoomEmail"),
        gmail_email=keys.get("GMAIL_EMAIL"),
        gmail_app_password=keys.get("GMAIL_APP_PASSWORD"),
        google_maps_api_key=keys.get("GOOGLE_MAP_API_KEY")
    )

    print("=" * 60)
    print("Unified Multi-Tool Agent (Notion + Zoom + Email + Google Maps)")
    print("=" * 60)
    print("\nExample queries:")
    print("  - What tasks do I have this week?")
    print("  - Schedule a Zoom meeting for COMP4901 project on Dec 10 at 3pm")
    print("  - Send email to professor about extending the deadline for comp4901 asm")
    print("  - Get my comp4901 notes and email them to my teammate")
    print("  - Find restaurants near HKUST and get directions")
    print("  - Get my tasks with locations and email directions to my teammate")
    print("\nType 'quit' to exit.\n")

    # Trajectory counter for unique filenames
    trajectory_count = 0

    # Ensure results directory exists (relative to project root)
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue

            print("\nProcessing...")
            result = agent.run(user_input)

            # Show tools used
            tools_used = [s['tool'] for s in result['steps']]
            unique_tools = list(set(tools_used))
            if result["steps"]:
                print(f"\n[Tools used: {', '.join(tools_used)}]")
                print(f"[Steps: {len(result['steps'])}, Unique tools: {len(unique_tools)}]")

            print(f"\nAgent: {result['final_response']}\n")

            # Save trajectory to JSON file
            trajectory_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"trajectory_{timestamp}_{trajectory_count}.json")

            trajectory_output = {
                "timestamp": datetime.now().isoformat(),
                "query": user_input,
                "steps": result["steps"],
                "final_response": result["final_response"],
                "tools_used": tools_used,
                "unique_tools": unique_tools,
                "step_count": len(result["steps"]),
                "tool_count": len(unique_tools)
            }

            import json
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(trajectory_output, f, indent=2, default=str, ensure_ascii=False)

            print(f"[Trajectory saved: {filename}]\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

