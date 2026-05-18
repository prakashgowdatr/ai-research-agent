# tools/report_tool.py
# Generates a structured markdown research report from collected findings.
# Uses the same LCEL chain pattern as summarizer_tool.py: prompt | llm | parser

from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import get_llm
from core.prompts import REPORT_PROMPT, REPORT_TOOL_DESCRIPTION


# ── Build the Report Chain ─────────────────────────────────────────────────────

def build_report_chain():
    """
    Constructs the LCEL report generation chain.

    Flow: {"query": ..., "findings": ...} → PromptTemplate → LLM → StrOutputParser
    Returns a runnable chain that produces a markdown report string.
    """
    prompt = PromptTemplate(
        template=REPORT_PROMPT,
        input_variables=["query", "findings"],
    )

    llm = get_llm()
    output_parser = StrOutputParser()

    return prompt | llm | output_parser


# Build once at module load — reused across all report generation calls.
_report_chain = build_report_chain()


# ── Report Generation Function ─────────────────────────────────────────────────

def generate_report(input_text: str) -> str:
    """
    Generates a structured research report from findings.

    The agent passes input in one of two formats:
    1. "QUERY: the query | FINDINGS: the collected findings"  ← preferred
    2. Just raw findings text                                  ← fallback

    Args:
        input_text: findings string from the agent

    Returns:
        Formatted markdown research report as a string
    """
    # Parse query and findings from the agent's input
    if "QUERY:" in input_text and "FINDINGS:" in input_text:
        parts = input_text.split("| FINDINGS:", 1)
        query = parts[0].replace("QUERY:", "").strip()
        findings = parts[1].strip()
    else:
        # Fallback: treat entire input as findings
        query = "Research Report"
        findings = input_text

    # Truncate findings if too long — report prompt + findings must fit in context
    if len(findings) > 6000:
        findings = findings[:6000] + "\n...[truncated]"
        print(f"[Report] Findings truncated to 6000 chars")

    print(f"[Report] Generating report for query: '{query[:60]}'")
    print(f"[Report] Findings length: {len(findings)} chars")

    try:
        report = _report_chain.invoke({
            "query": query,
            "findings": findings,
        })
        print(f"[Report] Generated {len(report)} char report")
        return report

    except Exception as e:
        return f"Report generation failed: {str(e)}"


# ── LangChain Tool Object ──────────────────────────────────────────────────────

report_tool = Tool(
    name="generate_report",
    description=REPORT_TOOL_DESCRIPTION,
    func=generate_report,
)