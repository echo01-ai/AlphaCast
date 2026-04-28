from dotenv import load_dotenv
import os
from pydantic_ai import Agent

load_dotenv()

model = os.getenv("PYA_MODEL")
if not model:
    raw = os.getenv("MODEL")
    model = f"openai:{raw}" if raw else None

base = os.getenv("OPENAI_BASE_URL")
if base and not os.getenv("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = base

if not model:
    raise RuntimeError("MODEL/PYA_MODEL is not configured in .env")

agent = Agent(model, instructions="Reply with exactly OK.")
result = agent.run_sync("Reply with exactly OK.")

print("model:", model)
print("output:", result.output)