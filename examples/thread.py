"""Thread workflow example with async runs."""

import os
from dotenv import load_dotenv
from pixigpt import Client

# Load .env
load_dotenv()

api_key = os.getenv("PIXIGPT_API_KEY")
base_url = os.getenv("PIXIGPT_BASE_URL")
assistant_id = os.getenv("DEFAULT_ASSISTANT_ID")

# Create client
client = Client(api_key, base_url)

# 1. Create thread
thread = client.create_thread()
print(f"Created thread: {thread.id}")

# 2. Add user message
msg = client.create_message(thread.id, "user", "What's the capital of France? Answer in one word.")
print(f"Added message: {msg.id}")

# 3. Create run
run = client.create_run(thread.id, assistant_id, enable_thinking=True)
print(f"Created run: {run.id} (status: {run.status})")

# 4. Wait for completion
print("Waiting for run to complete...")
completed_run = client.wait_for_run(thread.id, run.id)
print(f"Run completed: {completed_run.status}")

# 5. Get messages
messages = client.list_messages(thread.id, limit=10)

# Print conversation
print("\n=== Conversation ===")
for msg in reversed(messages):
    if msg.content:
        content = msg.content[0].text["value"]
        print(f"{msg.role}: {content}")

        if msg.reasoning_content:
            print(f"  [Reasoning: {msg.reasoning_content[:100]}...]")
