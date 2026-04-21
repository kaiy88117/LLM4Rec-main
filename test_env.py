from dotenv import load_dotenv
import os

load_dotenv()

print("✅ OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("✅ OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE"))
