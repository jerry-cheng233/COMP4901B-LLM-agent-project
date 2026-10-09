"""
Email-Sending LLM Agent.

This module implements an agent that uses DeepSeek LLM to parse email instructions
and send emails via Gmail SMTP. The agent:
1. Parses the instruction to extract recipient, subject, and body
2. Sends the email using stored sender credentials
3. Returns a trajectory with the details and status

Termination: Single step (no loop needed).
"""

import os
from typing import Dict, Any
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText

# Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_TOKENS = 1024

def send_gmail(sender: str, password: str, recipient: str, subject: str, body: str) -> str:
    """
    Send an email via Gmail SMTP.

    Args:
        sender: Sender email (Gmail)
        password: App password
        recipient: Recipient email (can be comma-separated for multiple)
        subject: Email subject
        body: Email body

    Returns:
        Status message (success or error)
    """
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipient.split(','), msg.as_string())
        return "Email sent successfully"
    except Exception as e:
        return f"Error sending email: {str(e)}"

class EmailAgent:
    """
    LLM agent for sending emails based on natural language instructions.
    
    Uses DeepSeek to parse instruction into to, subject, body, then sends via Gmail.
    """
    
    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-chat",
        sender_email: str = "",
        app_password: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        # Fetch from env if not provided explicitly
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not sender_email:
            sender_email = os.environ.get("GMAIL_EMAIL", "")
        if not app_password:
            app_password = os.environ.get("GMAIL_APP_PASSWORD", "")

        if not sender_email or not app_password:
            raise ValueError("GMAIL_EMAIL and GMAIL_APP_PASSWORD must be set in environment variables or passed explicitly.")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY must be set in environment variables or passed explicitly.")

        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_tokens = max_tokens
        self.sender_email = sender_email
        self.app_password = app_password
        
    def _get_system_prompt(self) -> str:
        """Return the system prompt for parsing email instructions."""
        return """You are an email composition assistant. From the user's instruction, extract:
- TO: recipient email(s) (comma-separated if multiple)
- SUBJECT: short, clear subject line
- BODY: polite, complete email body (include greetings/sign-offs if appropriate)

Output EXACTLY in this format:
TO: email1@example.com,email2@example.com
SUBJECT: Your Subject Here
BODY: Full email body here.

Make the body professional and concise. If no subject provided, infer one.
If multiple recipients, use comma-separated.
If instruction implies an excuse or specific content, generate appropriately."""

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract to, subject, body."""
        parsed = {"to": "", "subject": "", "body": ""}
        lines = response.strip().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("TO:"):
                parsed["to"] = line.split(":", 1)[1].strip()
            elif line.startswith("SUBJECT:"):
                parsed["subject"] = line.split(":", 1)[1].strip()
            elif line.startswith("BODY:"):
                # Capture the first part after "BODY:"
                body_start = line.split(":", 1)[1].strip()
                # Collect all remaining lines as body continuation
                remaining = "\n".join(lines[i+1:]).strip()
                parsed["body"] = body_start + ("\n" + remaining if remaining else "")
                break  # Body is last, no need to continue parsing
        return parsed

    def run(self, instruction: str) -> Dict[str, Any]:
        """
        Run the agent to parse and send an email based on instruction.

        Args:
            instruction: Natural language instruction (e.g., "Send email to prof@ust.hk asking for extension because sick.")

        Returns:
            Dictionary containing:
            - final_status: Success or error message
            - trajectory: Dict with parsed to, subject, body
        """
        trajectory = {
            "instruction": instruction,
            "parsed": {},
            "final_status": ""
        }

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": instruction}
        ]

        # Get LLM response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens
        )

        assistant_message = response.choices[0].message.content
        parsed = self._parse_response(assistant_message)
        trajectory["parsed"] = parsed

        if not all(parsed.values()):
            trajectory["final_status"] = "Error: Incomplete parsing of to/subject/body"
            return trajectory

        # Send email
        status = send_gmail(
            self.sender_email,
            self.app_password,
            parsed["to"],
            parsed["subject"],
            parsed["body"]
        )
        trajectory["final_status"] = status

        return trajectory

if __name__ == "__main__":
    # Test the agent with a self-email
    try:
        agent = EmailAgent()
        test_instruction = f"Send a test email to {os.environ.get('GMAIL_EMAIL', 'unknown')} with subject 'Agent Test' and body 'This is a test from EmailAgent.'"
        result = agent.run(test_instruction)
        print("Test Result:", result)
    except ValueError as e:
        print(e)