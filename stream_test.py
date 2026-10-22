import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

print("🤖 Nova is thinking...\n")

for chunk in client.models.generate_content_stream(
    model="gemini-3.6-flash",
    contents="Explain artificial intelligence in a simple way."
):
    if chunk.text:
        print(chunk.text, end="", flush=True)

print("\n")