# memory/memory.py
# Provides session and persistent memory for the agent.

from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from core.llm import get_llm
from config import SQLITE_DB_PATH
import os
os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

def get_memory(session_id: str = "default") -> ConversationSummaryMemory:
    """
    Returns a ConversationSummaryMemory instance backed by SQLite.
    
    session_id: unique string per conversation (e.g. username, UUID).
    Passing the same session_id restores that conversation's history.
    """
    chat_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{SQLITE_DB_PATH}",
    )

    memory = ConversationSummaryMemory(
        llm=get_llm(),
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
    )

    return memory