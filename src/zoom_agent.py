"""
Zoom Agent - Task Management Agent with Zoom Integration for Part II.

Module: zoom_agent.py

This module implements a realistic agent that:
1. Fetches TODO tasks from README.md or other markdown files
2. Splits complex tasks into smaller subtasks using LLM
3. Distributes tasks across deadlines
4. Creates Zoom meetings via Zoom API
5. Retrieves meeting details

Tools:
- parse_tasks: Read and parse TODO tasks from markdown files
- split_task: Break down a complex task into smaller subtasks
- schedule_tasks: Distribute tasks across deadlines
- create_meeting: Create a Zoom meeting
- get_meeting: Get details of an existing Zoom meeting

Usage:
    from zoom_agent import ZoomAgent
    agent = ZoomAgent(api_key=..., zoom_client_id=..., ...)
    result = agent.run("Parse tasks and create a meeting")
"""

import os
import json
import re
import base64
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from openai import OpenAI

# ============================================================================
# Configuration
# ============================================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Zoom API Configuration
ZOOM_CLIENT_ID = os.environ.get("ZOOM_CLIENT_ID", "")
ZOOM_CLIENT_SECRET = os.environ.get("ZOOM_CLIENT_SECRET", "")
ZOOM_ACCOUNT_ID = os.environ.get("ZOOM_ACCOUNT_ID", "")
ZOOM_EMAIL = os.environ.get("ZOOM_EMAIL", "")
ZOOM_API_BASE_URL = "https://api.zoom.us/v2"
ZOOM_OAUTH_URL = "https://zoom.us/oauth/token"

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_STEPS = 10


# ============================================================================
# Zoom API Helper Functions
# ============================================================================

def get_zoom_access_token(
    client_id: str = ZOOM_CLIENT_ID,
    client_secret: str = ZOOM_CLIENT_SECRET,
    account_id: str = ZOOM_ACCOUNT_ID
) -> Optional[str]:
    """
    Get OAuth access token for Zoom API using Server-to-Server OAuth.
    
    Args:
        client_id: Zoom app client ID
        client_secret: Zoom app client secret
        account_id: Zoom account ID
        
    Returns:
        Access token string or None if failed
    """
    # Create base64 encoded credentials
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    data = {
        "grant_type": "account_credentials",
        "account_id": account_id
    }
    
    try:
        response = requests.post(ZOOM_OAUTH_URL, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        return token_data.get("access_token")
    except Exception as e:
        print(f"Error getting Zoom access token: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def create_zoom_meeting(
    topic: str,
    start_time: str,
    duration: int = 60,
    agenda: str = "",
    access_token: str = None,
    user_email: str = ZOOM_EMAIL
) -> Dict[str, Any]:
    """
    Create a Zoom meeting.
    
    Args:
        topic: Meeting topic/title
        start_time: Start time in ISO 8601 format (e.g., "2024-12-05T10:00:00Z")
        duration: Meeting duration in minutes
        agenda: Meeting agenda/description
        access_token: Zoom OAuth access token
        user_email: Email of the user to create meeting for
        
    Returns:
        Dictionary with meeting details or error information
    """
    if not access_token:
        access_token = get_zoom_access_token()
        if not access_token:
            return {"error": "Failed to get access token"}
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    meeting_data = {
        "topic": topic,
        "type": 2,  # Scheduled meeting
        "start_time": start_time,
        "duration": duration,
        "timezone": "Asia/Hong_Kong",
        "agenda": agenda,
        "settings": {
            "host_video": True,
            "participant_video": True,
            "join_before_host": True,
            "mute_upon_entry": False,
            "waiting_room": False
        }
    }
    
    try:
        url = f"{ZOOM_API_BASE_URL}/users/{user_email}/meetings"
        response = requests.post(url, headers=headers, json=meeting_data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        error_msg = f"Error creating Zoom meeting: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" Response: {e.response.text}"
        return {"error": error_msg}


def get_meeting_details(
    meeting_id: str,
    access_token: str = None
) -> Dict[str, Any]:
    """
    Get details of an existing Zoom meeting.

    Args:
        meeting_id: The Zoom meeting ID
        access_token: Zoom OAuth access token

    Returns:
        Dictionary with meeting details or error information
    """
    if not access_token:
        access_token = get_zoom_access_token()
        if not access_token:
            return {"error": "Failed to get access token"}

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        url = f"{ZOOM_API_BASE_URL}/meetings/{meeting_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        error_msg = f"Error getting meeting details: {e}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" Response: {e.response.text}"
        return {"error": error_msg}


# ============================================================================
# Task Management Tools
# ============================================================================

def parse_tasks_from_markdown(file_path: str = "README.md") -> Dict[str, Any]:
    """
    Parse TODO tasks from a markdown file.

    Looks for common task patterns:
    - [ ] Unchecked checkbox items
    - [x] Checked checkbox items
    - Numbered lists with tasks
    - Bullet points under "Tasks", "TODO", "To Do" sections

    Args:
        file_path: Path to the markdown file

    Returns:
        Dictionary with parsed tasks and metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tasks = []

        # Pattern 1: Checkbox items [ ] or [x]
        checkbox_pattern = r'[-*]\s*\[([ x])\]\s*(.+?)(?=\n|$)'
        for match in re.finditer(checkbox_pattern, content, re.MULTILINE):
            completed = match.group(1).lower() == 'x'
            task_text = match.group(2).strip()
            tasks.append({
                "text": task_text,
                "completed": completed,
                "type": "checkbox"
            })

        # Pattern 2: Numbered list items under Tasks/TODO sections
        sections_pattern = r'(?:#{1,3}\s*(?:Tasks?|TODO|To\s*Do)[^\n]*\n)((?:(?:\d+\.|\-|\*)\s+[^\n]+\n?)+)'
        for section_match in re.finditer(sections_pattern, content, re.IGNORECASE):
            section_content = section_match.group(1)
            item_pattern = r'(?:\d+\.|\-|\*)\s+(.+?)(?=\n|$)'
            for item_match in re.finditer(item_pattern, section_content):
                task_text = item_match.group(1).strip()
                # Avoid duplicates from checkbox pattern
                if not any(t["text"] == task_text for t in tasks):
                    tasks.append({
                        "text": task_text,
                        "completed": False,
                        "type": "list_item"
                    })

        # Pattern 3: Look for parenthetical point items like "(30 Points)"
        points_pattern = r'\d+\.\s*\((\d+)\s*[Pp]oints?\)\s*(.+?)(?=\n\d+\.\s*\(|$)'
        for match in re.finditer(points_pattern, content, re.DOTALL):
            points = int(match.group(1))
            task_text = match.group(2).strip()
            # Clean up the task text - take first sentence or line
            task_text = task_text.split('\n')[0].strip()
            if task_text and not any(t["text"] == task_text for t in tasks):
                tasks.append({
                    "text": task_text,
                    "points": points,
                    "completed": False,
                    "type": "graded_task"
                })

        return {
            "file": file_path,
            "total_tasks": len(tasks),
            "tasks": tasks
        }
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"Error parsing file: {str(e)}"}


def split_task_into_subtasks(
    task: str,
    num_subtasks: int = 3,
    api_key: str = DEEPSEEK_API_KEY
) -> Dict[str, Any]:
    """
    Use LLM to split a complex task into smaller, manageable subtasks.

    Args:
        task: The main task to split
        num_subtasks: Target number of subtasks
        api_key: DeepSeek API key

    Returns:
        Dictionary with original task and list of subtasks
    """
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""Split the following task into {num_subtasks} smaller, actionable subtasks.
Each subtask should be:
- Specific and actionable
- Completable in a reasonable time frame
- Contributing to the main task completion

Task: {task}

Respond in JSON format:
{{
    "subtasks": [
        {{"id": 1, "title": "subtask title", "description": "brief description", "estimated_hours": 1}},
        ...
    ]
}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a project management assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )

        result_text = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        subtasks_data = json.loads(result_text)

        return {
            "original_task": task,
            "subtasks": subtasks_data.get("subtasks", []),
            "total_subtasks": len(subtasks_data.get("subtasks", []))
        }
    except Exception as e:
        return {"error": f"Error splitting task: {str(e)}"}


def schedule_tasks_with_deadlines(
    tasks: List[Dict[str, Any]],
    start_date: str = None,
    end_date: str = None,
    working_hours_per_day: int = 8
) -> Dict[str, Any]:
    """
    Distribute tasks across a timeline with deadlines.

    Args:
        tasks: List of tasks with estimated_hours
        start_date: Start date (ISO format, defaults to today)
        end_date: End date/deadline (ISO format, required)
        working_hours_per_day: Available working hours per day

    Returns:
        Dictionary with scheduled tasks
    """
    if not end_date:
        return {"error": "End date/deadline is required"}

    try:
        start = datetime.fromisoformat(start_date) if start_date else datetime.now()
        end = datetime.fromisoformat(end_date)

        if end <= start:
            return {"error": "End date must be after start date"}

        total_days = (end - start).days
        if total_days <= 0:
            total_days = 1

        # Calculate total hours needed
        total_hours = sum(t.get("estimated_hours", 2) for t in tasks)
        hours_per_day = total_hours / total_days

        scheduled_tasks = []
        current_date = start
        accumulated_hours = 0

        for i, task in enumerate(tasks):
            task_hours = task.get("estimated_hours", 2)
            accumulated_hours += task_hours

            # Calculate deadline based on accumulated hours
            days_needed = accumulated_hours / working_hours_per_day
            task_deadline = start + timedelta(days=int(days_needed) + 1)

            # Ensure deadline doesn't exceed end date
            if task_deadline > end:
                task_deadline = end

            scheduled_tasks.append({
                **task,
                "scheduled_deadline": task_deadline.isoformat(),
                "priority": i + 1
            })

        return {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "total_tasks": len(scheduled_tasks),
            "total_hours": total_hours,
            "scheduled_tasks": scheduled_tasks
        }
    except Exception as e:
        return {"error": f"Error scheduling tasks: {str(e)}"}


# ============================================================================
# Tool Definitions for Function Calling
# ============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "parse_tasks",
            "description": "Parse and extract TODO tasks from a markdown file like README.md",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the markdown file to parse (default: README.md)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "split_task",
            "description": "Split a complex task into smaller, manageable subtasks",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The main task to split into subtasks"
                    },
                    "num_subtasks": {
                        "type": "integer",
                        "description": "Target number of subtasks (default: 3)"
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_tasks",
            "description": "Distribute tasks across a timeline with deadlines",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "estimated_hours": {"type": "number"}
                            }
                        },
                        "description": "List of tasks with title and estimated_hours"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in ISO format (default: today)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date/deadline in ISO format"
                    }
                },
                "required": ["tasks", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_meeting",
            "description": "Create a Zoom meeting for team collaboration",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Meeting topic/title"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time in ISO 8601 format (e.g., 2024-12-05T10:00:00)"
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Meeting duration in minutes (default: 60)"
                    },
                    "agenda": {
                        "type": "string",
                        "description": "Meeting agenda/description"
                    }
                },
                "required": ["topic", "start_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_meeting",
            "description": "Get details of an existing Zoom meeting",
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "string",
                        "description": "The Zoom meeting ID"
                    }
                },
                "required": ["meeting_id"]
            }
        }
    }
]


# ============================================================================
# ZoomAgent Class
# ============================================================================

class ZoomAgent:
    """
    Zoom Agent - Task Management Agent with Zoom Integration.

    This agent can:
    1. Parse TODO tasks from markdown files
    2. Split complex tasks into subtasks
    3. Schedule tasks with deadlines
    4. Create Zoom meetings
    5. Get meeting details

    Uses DeepSeek LLM for reasoning and tool selection.
    """

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        zoom_client_id: str = ZOOM_CLIENT_ID,
        zoom_client_secret: str = ZOOM_CLIENT_SECRET,
        zoom_account_id: str = ZOOM_ACCOUNT_ID,
        zoom_email: str = ZOOM_EMAIL,
        model: str = "deepseek-chat",
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        self.api_key = api_key  # Store API key for tool calls
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        # Store Zoom credentials
        self.zoom_client_id = zoom_client_id
        self.zoom_client_secret = zoom_client_secret
        self.zoom_account_id = zoom_account_id
        self.zoom_email = zoom_email
        self._zoom_token = None

    def _get_zoom_token(self) -> Optional[str]:
        """Get or refresh Zoom access token."""
        if not self._zoom_token:
            self._zoom_token = get_zoom_access_token(
                self.zoom_client_id,
                self.zoom_client_secret,
                self.zoom_account_id
            )
        return self._zoom_token

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a helpful task management assistant with access to the following tools:

1. **parse_tasks** - Read and extract TODO tasks from markdown files (like README.md)
2. **split_task** - Break down complex tasks into smaller, actionable subtasks
3. **schedule_tasks** - Distribute tasks across a timeline with deadlines
4. **create_meeting** - Create Zoom meetings for team collaboration
5. **get_meeting** - Get details of existing Zoom meetings

Your job is to help users manage their project tasks efficiently. When given a request:
1. Analyze what needs to be done
2. Use the appropriate tools in sequence
3. Provide clear summaries of actions taken

Always explain your reasoning before calling tools. Be thorough but efficient."""

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return its result."""
        if tool_name == "parse_tasks":
            file_path = arguments.get("file_path", "README.md")
            return parse_tasks_from_markdown(file_path)

        elif tool_name == "split_task":
            task = arguments.get("task", "")
            num_subtasks = arguments.get("num_subtasks", 3)
            return split_task_into_subtasks(task, num_subtasks, api_key=self.api_key)

        elif tool_name == "schedule_tasks":
            tasks = arguments.get("tasks", [])
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")
            return schedule_tasks_with_deadlines(tasks, start_date, end_date)

        elif tool_name == "create_meeting":
            return create_zoom_meeting(
                topic=arguments.get("topic", "Meeting"),
                start_time=arguments.get("start_time"),
                duration=arguments.get("duration", 60),
                agenda=arguments.get("agenda", ""),
                access_token=self._get_zoom_token(),
                user_email=self.zoom_email
            )

        elif tool_name == "get_meeting":
            meeting_id = arguments.get("meeting_id")
            return get_meeting_details(meeting_id, self._get_zoom_token())

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def run(self, user_request: str) -> Dict[str, Any]:
        """
        Run the agent to handle a user request.

        Args:
            user_request: The user's task management request

        Returns:
            Dictionary containing:
            - final_response: The agent's final response
            - trajectory: List of steps taken (for demonstration)
        """
        trajectory = {
            "request": user_request,
            "steps": [],
            "final_response": ""
        }

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_request}
        ]

        step_count = 0

        while step_count < self.max_steps:
            step_count += 1

            # Get LLM response with tool calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=self.max_tokens
            )

            assistant_message = response.choices[0].message

            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                # Record thinking if present
                if assistant_message.content:
                    trajectory["steps"].append({
                        "step": step_count,
                        "type": "thinking",
                        "content": assistant_message.content
                    })

                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, arguments)

                    # Record the tool call
                    trajectory["steps"].append({
                        "step": step_count,
                        "type": "tool_call",
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": tool_result
                    })

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, default=str)
                    })

            else:
                # No tool calls - this is the final response
                final_response = assistant_message.content or ""
                trajectory["final_response"] = final_response
                trajectory["steps"].append({
                    "step": step_count,
                    "type": "final_response",
                    "content": final_response
                })
                break

        # If we hit max steps without a final response
        if not trajectory["final_response"]:
            trajectory["final_response"] = "Maximum steps reached. Please try a simpler request."

        trajectory["total_steps"] = step_count
        return trajectory


# Alias for backwards compatibility
TaskAgent = ZoomAgent
