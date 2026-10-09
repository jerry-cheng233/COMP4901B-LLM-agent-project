"""
Notion API Integration for LLM Tool Calling.

This module provides a Notion class that wraps the Notion API for use as an LLM tool.
The LLM agent can use this to create pages, query databases, update content, and more.

API Documentation: https://developers.notion.com/reference
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class Notion:
    """
    Notion API wrapper for LLM tool calling.
    
    Provides methods for:
    - Creating pages
    - Querying databases
    - Updating pages
    - Searching content
    - Managing blocks (content)
    """
    
    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Notion client.
        
        Args:
            api_key: Notion integration token. If not provided, reads from NOTION_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("NOTION_API_KEY", "")
        if not self.api_key:
            raise ValueError("Notion API key is required. Set NOTION_API_KEY environment variable or pass api_key parameter.")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION
        }
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make an API request to Notion."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=data)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "PATCH":
                response = requests.patch(url, headers=self.headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "details": response.json() if response.text else None}
        except Exception as e:
            return {"error": str(e)}
    
    def search(self, query: str, filter_type: str = None, page_size: int = 10) -> Dict[str, Any]:
        """
        Search for pages and databases in the workspace.
        
        Args:
            query: Search query string
            filter_type: Optional filter - "page" or "database"
            page_size: Number of results to return (max 100)
            
        Returns:
            Search results containing matching pages/databases
        """
        data = {
            "query": query,
            "page_size": min(page_size, 100)
        }
        if filter_type in ["page", "database"]:
            data["filter"] = {"property": "object", "value": filter_type}
        
        return self._make_request("POST", "search", data)
    
    def get_page(self, page_id: str) -> Dict[str, Any]:
        """
        Retrieve a page by its ID.
        
        Args:
            page_id: The ID of the page to retrieve
            
        Returns:
            Page object with properties and metadata
        """
        return self._make_request("GET", f"pages/{page_id}")
    
    def create_page(self, parent_id: str, title: str, content: List[Dict] = None, 
                    parent_type: str = "page", properties: Dict = None) -> Dict[str, Any]:
        """
        Create a new page in Notion.
        
        Args:
            parent_id: ID of the parent page or database
            title: Title of the new page
            content: Optional list of block objects for page content
            parent_type: "page" or "database"
            properties: Additional properties for database pages
            
        Returns:
            Created page object
        """
        if parent_type == "database":
            parent = {"database_id": parent_id}
            page_properties = properties or {}
            # Only add default title if no title property is already set
            # Check for common title property names
            has_title = any(key in page_properties for key in ["Name", "title", "Title", "Task name"])
            if not has_title:
                page_properties["Name"] = {"title": [{"text": {"content": title}}]}
        else:
            parent = {"page_id": parent_id}
            page_properties = {"title": {"title": [{"text": {"content": title}}]}}
        
        data = {
            "parent": parent,
            "properties": page_properties
        }
        
        if content:
            data["children"] = content
        
        return self._make_request("POST", "pages", data)
    
    def update_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a page's properties.
        
        Args:
            page_id: The ID of the page to update
            properties: Dictionary of properties to update
            
        Returns:
            Updated page object
        """
        return self._make_request("PATCH", f"pages/{page_id}", {"properties": properties})
    
    def archive_page(self, page_id: str) -> Dict[str, Any]:
        """
        Archive (soft delete) a page.
        
        Args:
            page_id: The ID of the page to archive
            
        Returns:
            Archived page object
        """
        return self._make_request("PATCH", f"pages/{page_id}", {"archived": True})
    
    def query_database(self, database_id: str, filter_obj: Dict = None, 
                       sorts: List[Dict] = None, page_size: int = 100) -> Dict[str, Any]:
        """
        Query a database with optional filters and sorts.
        
        Args:
            database_id: The ID of the database to query
            filter_obj: Optional filter object
            sorts: Optional list of sort objects
            page_size: Number of results per page
            
        Returns:
            Query results with matching pages
        """
        data = {"page_size": min(page_size, 100)}
        if filter_obj:
            data["filter"] = filter_obj
        if sorts:
            data["sorts"] = sorts
        
        return self._make_request("POST", f"databases/{database_id}/query", data)

    def get_database(self, database_id: str) -> Dict[str, Any]:
        """
        Retrieve a database by its ID.

        Args:
            database_id: The ID of the database

        Returns:
            Database object with schema and metadata
        """
        return self._make_request("GET", f"databases/{database_id}")

    def get_block_children(self, block_id: str, page_size: int = 100) -> Dict[str, Any]:
        """
        Retrieve the children blocks of a block or page.

        Args:
            block_id: The ID of the block or page
            page_size: Number of results per page

        Returns:
            List of child blocks
        """
        return self._make_request("GET", f"blocks/{block_id}/children", {"page_size": page_size})

    def append_blocks(self, block_id: str, children: List[Dict]) -> Dict[str, Any]:
        """
        Append new blocks as children of a block or page.

        Args:
            block_id: The ID of the parent block or page
            children: List of block objects to append

        Returns:
            Response with appended blocks
        """
        return self._make_request("PATCH", f"blocks/{block_id}/children", {"children": children})

    def delete_block(self, block_id: str) -> Dict[str, Any]:
        """
        Delete a block.

        Args:
            block_id: The ID of the block to delete

        Returns:
            Deleted block object
        """
        return self._make_request("DELETE", f"blocks/{block_id}")

    # ========== TASK MANAGEMENT FUNCTIONS (for To-do Lists calendar) ==========

    # Default database name to search for (will auto-discover ID)
    TODO_DATABASE_NAME = "To-do Lists"
    _todo_database_id_cache = None  # Cache for discovered database ID

    def _get_todo_database_id(self) -> str:
        """
        Auto-discover the To-do Lists database ID by searching for it by name.
        Caches the result to avoid repeated API calls.

        Returns:
            The database ID string

        Raises:
            ValueError if no matching database is found
        """
        if self._todo_database_id_cache:
            return self._todo_database_id_cache

        # Search for databases with the expected name
        result = self._make_request("POST", "search", {
            "query": self.TODO_DATABASE_NAME,
            "filter": {"property": "object", "value": "database"},
            "page_size": 10
        })

        if "error" in result:
            raise ValueError(f"Failed to search for database: {result['error']}")

        # Find the database with matching name
        for db in result.get("results", []):
            title_parts = db.get("title", [])
            title = "".join([t.get("plain_text", "") for t in title_parts])
            if title.lower() == self.TODO_DATABASE_NAME.lower():
                self._todo_database_id_cache = db["id"]
                return self._todo_database_id_cache

        # If exact match not found, use first result if available
        if result.get("results"):
            self._todo_database_id_cache = result["results"][0]["id"]
            return self._todo_database_id_cache

        raise ValueError(
            f"No database named '{self.TODO_DATABASE_NAME}' found. "
            "Make sure your Notion integration is connected to the database."
        )

    def get_tasks_by_date_range(self, start_date: str, end_date: str,
                                 include_done: bool = False,
                                 database_id: str = None) -> List[Dict]:
        """
        Get tasks within a date range from the To-do Lists calendar.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_done: Whether to include completed tasks
            database_id: Optional database ID (defaults to To-do Lists)

        Returns:
            List of tasks with their properties
        """
        db_id = database_id or self._get_todo_database_id()

        # Build filter for date range
        filter_obj = {
            "and": [
                {
                    "property": "Due",
                    "date": {"on_or_after": start_date}
                },
                {
                    "property": "Due",
                    "date": {"on_or_before": end_date}
                }
            ]
        }

        # Optionally exclude done tasks
        if not include_done:
            filter_obj["and"].append({
                "property": "Done",
                "checkbox": {"equals": False}
            })

        result = self.query_database(db_id, filter_obj=filter_obj,
                                     sorts=[{"property": "Due", "direction": "ascending"}])

        return self._parse_tasks(result)

    def get_tasks_for_today(self, include_done: bool = False) -> List[Dict]:
        """Get all tasks due today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_tasks_by_date_range(today, today, include_done)

    def get_tasks_for_tomorrow(self, include_done: bool = False) -> List[Dict]:
        """Get all tasks due tomorrow."""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        return self.get_tasks_by_date_range(tomorrow, tomorrow, include_done)

    def get_tasks_next_week(self, include_done: bool = False) -> List[Dict]:
        """Get all tasks for the next 7 days."""
        today = datetime.now()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        return self.get_tasks_by_date_range(start_date, end_date, include_done)

    def get_pending_tasks(self, database_id: str = None) -> List[Dict]:
        """Get all pending (not done) tasks sorted by due date."""
        db_id = database_id or self._get_todo_database_id()

        filter_obj = {
            "property": "Done",
            "checkbox": {"equals": False}
        }

        result = self.query_database(db_id, filter_obj=filter_obj,
                                     sorts=[{"property": "Due", "direction": "ascending"}])

        return self._parse_tasks(result)

    def get_overdue_tasks(self, database_id: str = None) -> List[Dict]:
        """Get all overdue tasks (past due date and not done)."""
        db_id = database_id or self._get_todo_database_id()
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        filter_obj = {
            "and": [
                {
                    "property": "Due",
                    "date": {"on_or_before": yesterday}
                },
                {
                    "property": "Done",
                    "checkbox": {"equals": False}
                }
            ]
        }

        result = self.query_database(db_id, filter_obj=filter_obj,
                                     sorts=[{"property": "Due", "direction": "ascending"}])

        return self._parse_tasks(result)

    def add_task(self, task_name: str, due_date: str, priority: str = None,
                 status: str = "Not Started", database_id: str = None) -> Dict[str, Any]:
        """
        Add a new task to the To-do Lists calendar.

        Args:
            task_name: Name of the task
            due_date: Due date in YYYY-MM-DD format
            priority: Priority level ("High", "Medium", "Low") or None
            status: Status ("Not Started", "In Progress", "Done")
            database_id: Optional database ID (defaults to To-do Lists)

        Returns:
            Created task/page object
        """
        db_id = database_id or self._get_todo_database_id()

        properties = {
            "Task name": {
                "title": [{"text": {"content": task_name}}]
            },
            "Due": {
                "date": {"start": due_date}
            },
            "Done": {
                "checkbox": False
            }
        }

        # Add priority if specified
        if priority and priority in ["High", "Medium", "Low"]:
            properties["Priority"] = {
                "select": {"name": priority}
            }

        # Add status if specified
        if status:
            properties["Status"] = {
                "status": {"name": status}
            }

        return self.create_page(
            parent_id=db_id,
            title=task_name,
            parent_type="database",
            properties=properties
        )

    def mark_task_done(self, page_id: str) -> Dict[str, Any]:
        """Mark a task as done."""
        return self.update_page(page_id, {"Done": {"checkbox": True}})

    def mark_task_undone(self, page_id: str) -> Dict[str, Any]:
        """Mark a task as not done."""
        return self.update_page(page_id, {"Done": {"checkbox": False}})

    def update_task_priority(self, page_id: str, priority: str) -> Dict[str, Any]:
        """Update task priority (High, Medium, Low)."""
        return self.update_page(page_id, {"Priority": {"select": {"name": priority}}})

    def update_task_due_date(self, page_id: str, due_date: str) -> Dict[str, Any]:
        """Update task due date."""
        return self.update_page(page_id, {"Due": {"date": {"start": due_date}}})

    def get_task_details(self, page_id: str) -> Dict[str, Any]:
        """
        Get full details of a task including its content blocks.

        Returns:
            Dict with task properties and content
        """
        page = self.get_page(page_id)
        blocks = self.get_block_children(page_id)

        # Parse the page properties
        task_info = self._parse_single_task(page) if "error" not in page else page

        # Add content blocks
        if "error" not in blocks:
            content_text = []
            for block in blocks.get("results", []):
                block_type = block.get("type")
                if block_type and block_type in block:
                    rich_text = block[block_type].get("rich_text", [])
                    for text_obj in rich_text:
                        content_text.append(text_obj.get("plain_text", ""))
            task_info["content"] = "\n".join(content_text) if content_text else ""

        return task_info

    def search_notes(self, query: str) -> List[Dict]:
        """
        Search for notes/pages that might contain task details.

        Args:
            query: Search term (e.g., task name)

        Returns:
            List of matching pages with their content
        """
        results = self.search(query, filter_type="page", page_size=10)

        if "error" in results:
            return [results]

        pages = []
        for page in results.get("results", []):
            page_info = {
                "id": page.get("id"),
                "url": page.get("url"),
                "title": self._extract_title(page),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time")
            }
            pages.append(page_info)

        return pages

    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Get the full content of a page including all text blocks.

        Args:
            page_id: ID of the page

        Returns:
            Dict with page title and full text content
        """
        # Get page metadata
        page = self.get_page(page_id)
        if "error" in page:
            return page

        title = self._extract_title(page)

        # Get all blocks (content) from the page
        blocks = self.get_block_children(page_id)
        if "error" in blocks:
            return {"title": title, "content": "", "error": blocks.get("error")}

        # Extract text from all blocks
        content_parts = []
        for block in blocks.get("results", []):
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            # Handle different block types
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3",
                             "bulleted_list_item", "numbered_list_item", "to_do", "quote"]:
                rich_text = block_data.get("rich_text", [])
                text = "".join([t.get("plain_text", "") for t in rich_text])
                if text:
                    if block_type.startswith("heading"):
                        content_parts.append(f"\n## {text}\n")
                    elif block_type == "bulleted_list_item":
                        content_parts.append(f"• {text}")
                    elif block_type == "numbered_list_item":
                        content_parts.append(f"- {text}")
                    elif block_type == "to_do":
                        checked = "✓" if block_data.get("checked") else "○"
                        content_parts.append(f"{checked} {text}")
                    else:
                        content_parts.append(text)
            elif block_type == "code":
                rich_text = block_data.get("rich_text", [])
                code = "".join([t.get("plain_text", "") for t in rich_text])
                lang = block_data.get("language", "")
                content_parts.append(f"```{lang}\n{code}\n```")

        return {
            "id": page_id,
            "title": title,
            "url": page.get("url", ""),
            "content": "\n".join(content_parts)
        }

    def add_note_to_task(self, page_id: str, note_text: str) -> Dict[str, Any]:
        """Add a note/description to a task."""
        blocks = [self.create_paragraph_block(note_text)]
        return self.append_blocks(page_id, blocks)

    def _parse_tasks(self, query_result: Dict) -> List[Dict]:
        """Parse query results into simplified task list."""
        if "error" in query_result:
            return [query_result]

        tasks = []
        for page in query_result.get("results", []):
            tasks.append(self._parse_single_task(page))

        return tasks

    def _parse_single_task(self, page: Dict) -> Dict:
        """Parse a single page into a simplified task dict."""
        props = page.get("properties", {})

        # Extract task name
        task_name = self._extract_title(page)

        # Extract due date
        due_date = None
        due_prop = props.get("Due", {}).get("date")
        if due_prop:
            due_date = due_prop.get("start")

        # Extract done status
        done = props.get("Done", {}).get("checkbox", False)

        # Extract priority
        priority = None
        priority_prop = props.get("Priority", {}).get("select")
        if priority_prop:
            priority = priority_prop.get("name")

        # Extract status
        status = None
        status_prop = props.get("Status", {}).get("status")
        if status_prop:
            status = status_prop.get("name")

        return {
            "id": page.get("id"),
            "task_name": task_name,
            "due_date": due_date,
            "done": done,
            "priority": priority,
            "status": status,
            "url": page.get("url")
        }

    def _extract_title(self, page: Dict) -> str:
        """Extract title from a page object."""
        props = page.get("properties", {})

        # Try common title property names
        for title_key in ["Task name", "Name", "title", "Title"]:
            if title_key in props:
                title_prop = props[title_key]
                if title_prop.get("type") == "title":
                    title_list = title_prop.get("title", [])
                    if title_list:
                        return title_list[0].get("plain_text", "")

        return "Untitled"

    # Helper methods for creating common block types
    @staticmethod
    def create_paragraph_block(text: str) -> Dict:
        """Create a paragraph block."""
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    @staticmethod
    def create_heading_block(text: str, level: int = 1) -> Dict:
        """Create a heading block (level 1, 2, or 3)."""
        heading_type = f"heading_{min(max(level, 1), 3)}"
        return {
            "object": "block",
            "type": heading_type,
            heading_type: {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    @staticmethod
    def create_bulleted_list_block(text: str) -> Dict:
        """Create a bulleted list item block."""
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    @staticmethod
    def create_numbered_list_block(text: str) -> Dict:
        """Create a numbered list item block."""
        return {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    @staticmethod
    def create_todo_block(text: str, checked: bool = False) -> Dict:
        """Create a to-do block."""
        return {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "checked": checked
            }
        }

    @staticmethod
    def create_code_block(code: str, language: str = "plain text") -> Dict:
        """Create a code block."""
        return {
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": code}}],
                "language": language
            }
        }

    def get_tool_definitions(self) -> List[Dict]:
        """
        Return tool definitions for LLM function calling (task management focused).

        Returns:
            List of tool definitions compatible with OpenAI-style function calling
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_tasks_today",
                    "description": "Get all tasks due today from the To-do Lists calendar",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tasks_tomorrow",
                    "description": "Get all tasks due tomorrow from the To-do Lists calendar",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tasks_next_week",
                    "description": "Get all tasks for the next 7 days from the To-do Lists calendar",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tasks_by_date_range",
                    "description": "Get tasks within a specific date range",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                            "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                            "include_done": {"type": "boolean", "description": "Include completed tasks (default: false)"}
                        },
                        "required": ["start_date", "end_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pending_tasks",
                    "description": "Get all pending (not done) tasks sorted by due date",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_overdue_tasks",
                    "description": "Get all overdue tasks (past due date and not done)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_task",
                    "description": "Add a new task to the To-do Lists calendar",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "Name of the task"},
                            "due_date": {"type": "string", "description": "Due date in YYYY-MM-DD format"},
                            "priority": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Priority level"}
                        },
                        "required": ["task_name", "due_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_done",
                    "description": "Mark a task as completed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "ID of the task to mark as done"}
                        },
                        "required": ["page_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_task_details",
                    "description": "Get full details of a task including its content/notes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "ID of the task"}
                        },
                        "required": ["page_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_notes",
                    "description": "Search for notes/pages that might contain task details. Returns page IDs and titles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search term (e.g., task name or keyword)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_content",
                    "description": "Get the full text content of a Notion page. Use this after search_notes to read a page's content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "ID of the page to read"}
                        },
                        "required": ["page_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_note_to_task",
                    "description": "Add a note/description to an existing task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "ID of the task"},
                            "note_text": {"type": "string", "description": "Note text to add"}
                        },
                        "required": ["page_id", "note_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_task_priority",
                    "description": "Update the priority of a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "ID of the task"},
                            "priority": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "New priority"}
                        },
                        "required": ["page_id", "priority"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_task_due_date",
                    "description": "Update the due date of a task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_id": {"type": "string", "description": "ID of the task"},
                            "due_date": {"type": "string", "description": "New due date in YYYY-MM-DD format"}
                        },
                        "required": ["page_id", "due_date"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """
        Execute a tool call from the LLM.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Tool execution result
        """
        tool_map = {
            "get_tasks_today": lambda: self.get_tasks_for_today(),
            "get_tasks_tomorrow": lambda: self.get_tasks_for_tomorrow(),
            "get_tasks_next_week": lambda: self.get_tasks_next_week(),
            "get_tasks_by_date_range": lambda: self.get_tasks_by_date_range(
                arguments["start_date"],
                arguments["end_date"],
                arguments.get("include_done", False)
            ),
            "get_pending_tasks": lambda: self.get_pending_tasks(),
            "get_overdue_tasks": lambda: self.get_overdue_tasks(),
            "add_task": lambda: self.add_task(
                arguments["task_name"],
                arguments["due_date"],
                arguments.get("priority")
            ),
            "mark_task_done": lambda: self.mark_task_done(arguments["page_id"]),
            "get_task_details": lambda: self.get_task_details(arguments["page_id"]),
            "search_notes": lambda: self.search_notes(arguments["query"]),
            "get_page_content": lambda: self.get_page_content(arguments["page_id"]),
            "add_note_to_task": lambda: self.add_note_to_task(
                arguments["page_id"],
                arguments["note_text"]
            ),
            "update_task_priority": lambda: self.update_task_priority(
                arguments["page_id"],
                arguments["priority"]
            ),
            "update_task_due_date": lambda: self.update_task_due_date(
                arguments["page_id"],
                arguments["due_date"]
            )
        }

        if tool_name in tool_map:
            return tool_map[tool_name]()
        else:
            return {"error": f"Unknown tool: {tool_name}"}


class NotionAgent:
    """
    LLM Agent for managing Notion tasks using natural language.

    This agent uses DeepSeek LLM with Notion API tools to:
    1. Query tasks from the To-do Lists calendar
    2. Search for notes/details about tasks
    3. Add new tasks with priorities and due dates
    4. Update and manage existing tasks
    """

    def __init__(
        self,
        deepseek_api_key: str,
        notion_api_key: str,
        model: str = "deepseek-chat",
        max_steps: int = 5,
        max_tokens: int = 1024
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        self.notion = Notion(api_key=notion_api_key)
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.tools = self.notion.get_tool_definitions()

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""You are a helpful task management assistant with access to the user's Notion To-do Lists.
Today's date is {today}.

You can help the user:
1. Query tasks (today, tomorrow, next week, custom date ranges, pending, overdue)
2. Search for notes/details about specific tasks
3. Add new tasks with due dates and priorities
4. Mark tasks as done
5. Update task priorities or due dates
6. Add notes to tasks

When the user asks about tasks, use the appropriate tool to fetch the information.
When adding tasks, extract the task name, due date, and priority from the user's request.
Convert relative dates like "tomorrow", "next Monday", "Dec 15" to YYYY-MM-DD format.

Always provide clear, helpful responses summarizing the results.
If a task has an ID, include it so the user can reference it for updates."""

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
                    result = self.notion.execute_tool(tool_name, arguments)

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

        # If we exhausted steps, get final response
        if not trajectory["final_response"]:
            trajectory["final_response"] = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

        return trajectory

    def chat(self, user_message: str) -> str:
        """
        Simple chat interface - returns just the response text.

        Args:
            user_message: The user's request

        Returns:
            The agent's response string
        """
        result = self.run(user_message)
        return result["final_response"]
