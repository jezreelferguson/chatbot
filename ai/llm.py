import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import json

from openai import chat

from openai import chat
load_dotenv()

with open('data/profile.json', 'r') as file:
    profile = json.load(file)
print("Profile loaded successfully.", profile)

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b")

async def ask_ai(question: str):
 prompt = f"""
You are AnokyeBot, the AI assistant created by Anokye Ferguson Adu, you are an AI assistant to {profile['name']}.
Your role:
- Answer questions ONLY using the provided profile data.
- Be concise, professional, and accurate.
- If the information is unavailable, say:
  "I don't have enough information to answer that."

Rules:
- Never invent information.
- Never assume facts not present in the profile.
- Keep responses clean and well formatted.
- Use bullet points when appropriate.
- Tone: friendly, intelligent, modern


PROFILE DATA:
{json.dumps(profile, indent=2)}

QUESTION:
{question}
"""

 response = await llm.ainvoke(prompt)
 return response.content