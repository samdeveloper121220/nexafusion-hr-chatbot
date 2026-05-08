# utils/memory.py - Simplified for compatibility
from langchain_core.messages import HumanMessage, AIMessage

# Simple in-memory chat history (works with current LangChain)
chat_history = []

def get_memory():
    return chat_history

def add_to_memory(role: str, content: str):
    if role == "user":
        chat_history.append(HumanMessage(content=content))
    else:
        chat_history.append(AIMessage(content=content))

# Clear memory function
def clear_memory():
    global chat_history
    chat_history = []