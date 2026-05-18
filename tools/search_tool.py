# tools/search_tool.py
# Provides web search capability to the agent.
# Primary: Tavily (structured, LLM-optimised results)
# Fallback: DuckDuckGo (no API key, unlimited, less structured)

from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from duckduckgo_search import DDGS
from config import TAVILY_API_KEY, SEARCH_MAX_RESULTS
from core.prompts import SEARCH_TOOL_DESCRIPTION


# ── Tavily Search ──────────────────────────────────────────────────────────────

def tavily_search(query: str) -> str:
    """
    Search using Tavily API — returns clean, structured results.
    Tavily is built specifically for LLM agents: it returns
    pre-cleaned text snippets rather than raw HTML.

    Args:
        query: the search query string from the agent

    Returns:
        formatted string of search results, or error message
    """
    try:
        # TavilySearchResults is a LangChain-native Tavily wrapper.
        # max_results controls how many pages we fetch per search.
        searcher = TavilySearchResults(
            max_results=SEARCH_MAX_RESULTS,
            tavily_api_key=TAVILY_API_KEY,
        )

        # .invoke() runs the search and returns a list of dicts.
        # Each dict has: 'url', 'content' (clean text snippet), 'title'
        results = searcher.invoke(query)

        if not results:
            return "No results found for this query."

        # Format results into a readable string for the agent.
        # The agent receives this string as its "observation".
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")
            formatted.append(f"[{i}] {title}\nURL: {url}\n{content}\n")

        return "\n".join(formatted)

    except Exception as e:
        # If Tavily fails for any reason, fall through to DuckDuckGo.
        print(f"Tavily search failed: {e}. Falling back to DuckDuckGo...")
        return duckduckgo_search(query)


# ── DuckDuckGo Fallback ────────────────────────────────────────────────────────

def duckduckgo_search(query: str) -> str:
    """
    Fallback search using DuckDuckGo — no API key required.
    Less structured than Tavily but always available.

    Args:
        query: the search query string

    Returns:
        formatted string of search results, or error message
    """
    try:
        # DDGS() is the DuckDuckGo search client.
        # text() returns a generator of result dicts.
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=SEARCH_MAX_RESULTS,
                )
            )

        if not results:
            return "No results found via DuckDuckGo either."

        # DuckDuckGo returns: 'title', 'href' (URL), 'body' (snippet)
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", "No URL")
            content = result.get("body", "No content")
            formatted.append(f"[{i}] {title}\nURL: {url}\n{content}\n")

        return "\n".join(formatted)

    except Exception as e:
        return f"Both search methods failed. Error: {str(e)}"


# ── Smart Search: Tavily First, DuckDuckGo on Failure ─────────────────────────

def smart_search(query: str) -> str:
    """
    The function the agent actually calls.
    Tries Tavily first; DuckDuckGo is the automatic fallback inside tavily_search().
    This function is the single entry point — the agent never calls
    tavily_search() or duckduckgo_search() directly.

    Args:
        query: search query from the agent

    Returns:
        formatted search results as a string
    """
    print(f"\n[Search] Query: {query}")      # visible in terminal during dev
    result = tavily_search(query)
    print(f"[Search] Got {len(result)} chars of results")
    return result


# ── LangChain Tool Object ──────────────────────────────────────────────────────

# This is what gets imported and registered with the agent in agent.py.
# The agent sees the name and description — it never calls the functions directly.
search_tool = Tool(
    name="web_search",
    description=SEARCH_TOOL_DESCRIPTION,
    func=smart_search,
)