#!/usr/bin/env python3
"""
Gmail AI Agent - Send emails through Gmail using Claude as the AI backbone.

Setup:
    1. Enable 2-Step Verification in your Google Account.
    2. Create an App Password at: https://myaccount.google.com/apppasswords
       (Google Account → Security → 2-Step Verification → App passwords)
    3. Set the following environment variables (or add them to a .env file):
         GMAIL_ADDRESS       your Gmail address
         GMAIL_APP_PASSWORD  the 16-character app password
         ANTHROPIC_API_KEY   your Anthropic API key

Usage:
    python gmail_agent.py
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Gmail tool
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful email assistant with access to Gmail.
When the user asks you to send an email, use the send_gmail tool.
Always confirm the details of what was sent after each successful send.
Write professional, clear, and concise emails unless instructed otherwise.
"""


@beta_tool
def send_gmail(to: str, subject: str, body: str, is_html: bool = False) -> str:
    """Send an email through Gmail using SMTP.

    Args:
        to: Recipient email address (e.g. alice@example.com).
        subject: Email subject line.
        body: Email body content.
        is_html: Set to True if the body contains HTML markup. Defaults to plain text.
    """
    gmail_address = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to
    msg.attach(MIMEText(body, "html" if is_html else "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, app_password)
        server.sendmail(gmail_address, to, msg.as_string())

    return f"Email successfully sent to {to} with subject: '{subject}'"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(user_prompt: str, messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Run one turn of the agent loop.

    Returns the assistant's text response and the updated messages list.
    """
    client = anthropic.Anthropic()

    messages.append({"role": "user", "content": user_prompt})

    runner = client.beta.messages.tool_runner(
        model="claude-opus-4-6",
        max_tokens=4096,
        tools=[send_gmail],
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    # Iterate until the tool loop is done; keep the last message
    final_message = None
    for message in runner:
        final_message = message

    if final_message is None:
        return "(no response)", messages

    # Extract the assistant's text reply
    response_text = "".join(
        block.text for block in final_message.content if hasattr(block, "text")
    )

    # Persist the assistant turn as plain text so follow-up turns retain context
    messages.append({"role": "assistant", "content": response_text})

    return response_text, messages


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------


def main():
    # Load .env if present
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    # Validate required variables
    missing = [v for v in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "ANTHROPIC_API_KEY") if not os.getenv(v)]
    if missing:
        print(f"Error: missing environment variable(s): {', '.join(missing)}")
        print("See the module docstring for setup instructions.")
        raise SystemExit(1)

    print("Gmail AI Agent  (powered by Claude)")
    print("=" * 42)
    print("Describe the email you want to send, or type 'quit' to exit.\n")

    conversation: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response, conversation = run_agent(user_input, conversation)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
