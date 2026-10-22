import os
import pyttsx3
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
engine = pyttsx3.init()

voices = engine.getProperty("voices")

engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)
voices = engine.getProperty("voices")

for index, voice in enumerate(voices):
    print(index, voice.name)

SYSTEM_INSTRUCTION = """
You are a helpful AI voice assistant.

Your name is Nova.

Rules:
- Keep your answers short and conversational.
- Speak in simple language.
- Since your answers will be spoken aloud, avoid long paragraphs.
- Do not use unnecessary markdown.
- Be friendly and helpful.
- If the user asks a technical question, explain it step by step.
"""

previous_interaction_id = None

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if previous_interaction_id is None:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=user_input
        )

    else:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=user_input,
            previous_interaction_id=previous_interaction_id
        )

    print("AI:", interaction.output_text)

    previous_interaction_id = interaction.id