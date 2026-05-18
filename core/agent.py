# core/agent.py

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from core.llm import get_llm
from tools.search_tool import smart_search
from tools.summarizer_tool import summarize_content
from tools.report_tool import generate_report


def get_agent() -> AgentExecutor:
    llm = get_llm()

    tools = [
        Tool(
            name="web_search",
            func=smart_search,
            description="Search the internet for current information on any topic. Use this first when you need facts or recent data.",
        ),
        Tool(
            name="summarizer",
            func=summarize_content,
            description="Summarize and compress raw search results into clean bullet points. Use this after web_search to process the results.",
        ),
        Tool(
            name="report_generator",
            func=generate_report,
            description="Generate a structured markdown research report from summarized findings. Use this as the final step to produce the report.",
        ),
    ]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=15,
        max_execution_time=120,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )