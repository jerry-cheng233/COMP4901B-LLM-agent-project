"""
Demo script for NotionAgent - LLM-powered task management.

This shows how to connect the Notion API with DeepSeek LLM
to handle natural language requests for task management.
"""

import sys
sys.path.insert(0, 'src')

from notion import NotionAgent

# API Keys
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"  # Replace with your actual DeepSeek API key
NOTION_API_KEY = "YOUR_NOTION_API_KEY_HERE"  # Replace with your actual Notion API key

def main():
    # Initialize the agent
    agent = NotionAgent(
        deepseek_api_key=DEEPSEEK_API_KEY,
        notion_api_key=NOTION_API_KEY
    )
    
    print("=" * 60)
    print("NotionAgent Demo - Natural Language Task Management")
    print("=" * 60)
    print()
    
    # Example queries to demonstrate the agent
    test_queries = [
        "What tasks do I have today?",
        "Show me what's due this week",
        "What are my comp4901 related tasks?",
        # "Add a task: COMP4901B final project due on December 15, high priority",
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"User: {query}")
        print(f"{'─' * 60}")
        
        # Run the agent
        result = agent.run(query)
        
        # Print the steps taken (tools called)
        if result["steps"]:
            print("\n📋 Tools used:")
            for step in result["steps"]:
                print(f"  • {step['tool']}({step['arguments']})")
        
        # Print the final response
        print(f"\n🤖 Assistant: {result['final_response']}")
    
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60)
    
    # Interactive mode
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue
            
            response = agent.chat(user_input)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

