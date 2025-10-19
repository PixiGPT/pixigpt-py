"""Simple chat completion example."""

import os
from dotenv import load_dotenv
from pixigpt import Client, ChatCompletionRequest, Message

# Load .env
load_dotenv()

api_key = os.getenv("PIXIGPT_API_KEY")
base_url = os.getenv("PIXIGPT_BASE_URL")
assistant_id = os.getenv("DEFAULT_ASSISTANT_ID")

# Create client
client = Client(api_key, base_url)

# Send chat completion
response = client.create_chat_completion(
    ChatCompletionRequest(
        assistant_id=assistant_id,
        messages=[Message(role="user", content="Hello! What's your name?")],
        temperature=0.7,
        max_tokens=2000,
    )
)

# Print response
choice = response.choices[0]
print(f"Assistant: {choice.message.content}")

if choice.reasoning_content:
    print(f"\nReasoning: {choice.reasoning_content[:200]}...")

print(f"\nUsage: {response.usage.prompt_tokens} input + {response.usage.completion_tokens} output = {response.usage.total_tokens} total")
