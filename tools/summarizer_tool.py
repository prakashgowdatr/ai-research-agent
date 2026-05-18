# tools/summarizer_tool.py
# Compresses raw search results into clean bullet points using the LLM.
# Demonstrates the LCEL chain pattern: prompt | llm | output_parser

from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import get_llm
from core.prompts import SUMMARIZER_PROMPT, SUMMARIZER_TOOL_DESCRIPTION


# ── Build the Summarizer Chain ─────────────────────────────────────────────────

def build_summarizer_chain():
    """
    Constructs and returns the LCEL summarizer chain.

    The chain has three stages connected by the pipe operator |:
        PromptTemplate  →  LLM  →  StrOutputParser

    PromptTemplate: takes {query} and {content}, fills the prompt string
    LLM:            receives the filled prompt, returns an AIMessage object
    StrOutputParser: extracts the .content string from the AIMessage

    Returns:
        A runnable chain that accepts a dict and returns a string.
    """
    # PromptTemplate turns our string template into a LangChain-runnable object.
    # input_variables tells it which {placeholders} to expect.
    prompt = PromptTemplate(
        template=SUMMARIZER_PROMPT,
        input_variables=["query", "content"],
    )

    llm = get_llm()

    # StrOutputParser extracts the plain string from the LLM's AIMessage response.
    # Without it, the chain returns an AIMessage object instead of a string.
    output_parser = StrOutputParser()

    # The pipe operator | chains these three together into one runnable.
    # When you call .invoke() on the chain, data flows left to right:
    # dict → prompt fills it → LLM processes it → parser extracts string
    chain = prompt | llm | output_parser

    return chain


# ── Summarizer Function ────────────────────────────────────────────────────────

# Build the chain once at module load — reused for every summarizer call.
_summarizer_chain = build_summarizer_chain()


def summarize_content(input_text: str) -> str:
    """
    Summarizes raw content using the LLM chain.

    The agent passes input in one of two formats:
    1. "QUERY: the query | CONTENT: the raw text"   ← preferred, more focused
    2. Just raw text                                  ← fallback, generic summary

    Args:
        input_text: raw content string from the agent

    Returns:
        Clean bullet-point summary as a string
    """
    # Parse the input to extract query and content separately.
    # The agent may pass them combined or as raw content only.
    if "QUERY:" in input_text and "CONTENT:" in input_text:
        # Split on the separator the agent uses
        parts = input_text.split("| CONTENT:", 1)
        query = parts[0].replace("QUERY:", "").strip()
        content = parts[1].strip()
    else:
        # No query provided — summarize generically
        query = "general research"
        content = input_text

    # Truncate content to avoid hitting token limits.
    # At ~4 chars per token, 8000 chars ≈ 2000 tokens — safe for most models.
    if len(content) > 8000:
        content = content[:8000] + "\n...[truncated for length]"
        print(f"[Summarizer] Content truncated to 8000 chars")

    print(f"[Summarizer] Summarizing {len(content)} chars for query: '{query[:50]}'")

    try:
        # .invoke() runs the full chain: prompt filling → LLM → string output
        summary = _summarizer_chain.invoke({
            "query": query,
            "content": content,
        })
        print(f"[Summarizer] Produced {len(summary)} char summary")
        return summary

    except Exception as e:
        return f"Summarization failed: {str(e)}. Original content: {content[:500]}"


# ── LangChain Tool Object ──────────────────────────────────────────────────────

summarizer_tool = Tool(
    name="summarize_content",
    description=SUMMARIZER_TOOL_DESCRIPTION,
    func=summarize_content,
)