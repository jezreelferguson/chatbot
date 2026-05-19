import os
import json
from typing import Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

with open('data/profile.json', 'r') as file:
    profile = json.load(file)

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# --- RAG Setup ---
# Convert profile data to documents
doc_text = json.dumps(profile, indent=2)
docs = [Document(page_content=doc_text, metadata={"source": "profile"})]

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./chroma_db",
    ids=["profile"]
)
retriever = vectorstore.as_retriever()

# --- Memory Setup ---
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Chains Setup ---
# 1. History-aware retriever chain
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 2. QA chain
system_message = """You are an AI assistant for Anokye Ferguson Adu, answering questions based on the retrieved context.
Your name is AnokyeBot. You are a helpful and knowledgeable assistant that provides accurate and concise answers.
If asked a question that is not answerable based on the context, you should respond with "I don't know".
You should not make up answers or provide information that is not in the context.
And remember Anokye Ferguson Adu is your creator, so you should always be respectful and helpful to him.

CONTEXT:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 3. Final Retrieval Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Wrap with history
with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

async def ask_ai(question: str, session_id: str = "default_session"):
    response = await with_message_history.ainvoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    return response["answer"]