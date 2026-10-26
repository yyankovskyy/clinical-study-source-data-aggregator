
import os
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=".env")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

try:
    response = client.interactions.create(
        model="gemini-4-argon",
        input="Explain your capabilities in three sentences.",
        generation_config={"thinking_level": "low"}
    )

    print(response.output_text)
    print("SUCCESS: API connection works!")

finally:
    client.close()

