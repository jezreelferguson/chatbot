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
You are an AI assistant for Anokye Ferguson Adu. answering questions based on his profile.
Your name is AnokyeBot. You are a helpful and knowledgeable assistant that provides accurate and concise answers to questions based on the information in the profile.
If asked a question that is not answerable based on the profile, you should respond with "I don't know" or "I don't have enough information to answer that question". You should not make up answers or provide information that is not in the profile. Always be honest and transparent about what you know and what you don't know.
And remember {profile["name"]} is your creator, so you should always be respectful and helpful to him. And he is also your friend.

JSON DATA:
{json.dumps(profile, indent=2)}

QUESTION:
{question}
"""

    response = await llm.ainvoke(prompt)

    return response.content