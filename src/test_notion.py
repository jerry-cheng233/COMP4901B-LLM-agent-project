"""Test script for Notion API integration - Task Management Functions."""

import sys
sys.path.insert(0, 'src')

from notion import Notion
import json
from datetime import datetime, timedelta

# Load API key
NOTION_API_KEY = "YOUR_NOTION_API_KEY_HERE"  # Replace with your actual Notion API key

def print_tasks(tasks, title):
    """Helper to print tasks nicely."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    if not tasks:
        print("No tasks found.")
        return

    for i, task in enumerate(tasks, 1):
        if "error" in task:
            print(f"Error: {task}")
            continue

        done_marker = "✅" if task.get("done") else "⬜"
        priority = f"[{task.get('priority', 'No priority')}]" if task.get("priority") else ""
        print(f"{i}. {done_marker} {task.get('task_name')} - Due: {task.get('due_date')} {priority}")
        print(f"   ID: {task.get('id')}")

def main():
    notion = Notion(api_key=NOTION_API_KEY)

    # ========== TEST 1: Get tasks for today ==========
    print_tasks(notion.get_tasks_for_today(), "Tasks Due Today")

    # ========== TEST 2: Get tasks for tomorrow ==========
    print_tasks(notion.get_tasks_for_tomorrow(), "Tasks Due Tomorrow")

    # ========== TEST 3: Get tasks for next 7 days ==========
    print_tasks(notion.get_tasks_next_week(), "Tasks for Next 7 Days")

    # Skip listing all pending/overdue tasks for brevity
    # # ========== TEST 4: Get all pending (not done) tasks ==========
    # print_tasks(notion.get_pending_tasks(), "All Pending Tasks (Not Done)")
    #
    # # ========== TEST 5: Get overdue tasks ==========
    # print_tasks(notion.get_overdue_tasks(), "Overdue Tasks")

    # ========== TEST 6: Search for notes ==========
    print(f"\n{'=' * 60}")
    print("Search Results for 'comp4901'")
    print(f"{'=' * 60}")
    notes = notion.search_notes("comp4901")
    for note in notes:
        print(f"- {note.get('title')} (ID: {note.get('id')})")

    # ========== TEST 7: Add a new task ==========
    print(f"\n{'=' * 60}")
    print("Adding a new task: 'Test task from API'")
    print(f"{'=' * 60}")

    # Use tomorrow's date for the test task
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    result = notion.add_task(
        task_name="Test task from API",
        due_date=tomorrow,
        priority="High"
    )

    if "error" in result:
        print(f"Error adding task: {result}")
    else:
        print(f"✅ Task created successfully!")
        print(f"   ID: {result.get('id')}")
        print(f"   URL: {result.get('url')}")

        # Get the task details
        task_id = result.get("id")

        # ========== TEST 8: Add note to task ==========
        print(f"\n{'=' * 60}")
        print("Adding a note to the new task")
        print(f"{'=' * 60}")

        note_result = notion.add_note_to_task(
            task_id,
            "This is a test note added via the Notion API. It can include task details, descriptions, or any other information."
        )

        if "error" in note_result:
            print(f"Error adding note: {note_result}")
        else:
            print("✅ Note added successfully!")

        # ========== TEST 9: Get task details ==========
        print(f"\n{'=' * 60}")
        print("Getting full task details")
        print(f"{'=' * 60}")

        details = notion.get_task_details(task_id)
        print(json.dumps(details, indent=2, default=str))

        # ========== TEST 10: Mark task as done ==========
        print(f"\n{'=' * 60}")
        print("Marking the task as done")
        print(f"{'=' * 60}")

        done_result = notion.mark_task_done(task_id)
        if "error" in done_result:
            print(f"Error: {done_result}")
        else:
            print("✅ Task marked as done!")

    # ========== TEST 11: Custom date range query ==========
    print(f"\n{'=' * 60}")
    print("Tasks from Dec 1-15, 2025")
    print(f"{'=' * 60}")

    dec_tasks = notion.get_tasks_by_date_range("2025-12-01", "2025-12-15", include_done=True)
    print_tasks(dec_tasks, "December 1-15 Tasks (including done)")

if __name__ == "__main__":
    main()

