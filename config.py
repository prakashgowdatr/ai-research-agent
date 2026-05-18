# config.py
import os
from dotenv import load_dotenv

load_dotenv()

def _get(key: str) -> str:
    """Try Streamlit secrets first, fall back to environment variable."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

GROQ_API_KEY: str = _get("GROQ_API_KEY")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
TAVILY_API_KEY: str = _get("TAVILY_API_KEY")
LLM_TEMPERATURE: float = 0.3
LLM_MAX_TOKENS: int = 2048
SQLITE_DB_PATH: str = "data/research.db"
SEARCH_MAX_RESULTS: int = 5 