# app.py
import streamlit as st
from core.agent import get_agent
from core.llm import get_llm
from memory.memory import get_memory
from langchain.schema import HumanMessage, SystemMessage

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")
st.title("🔬 AI Research Assistant")
st.caption("Powered by Groq (Llama 3.3) · LangChain · Tavily")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = get_agent()
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Settings")
    show_steps = st.toggle("Show agent reasoning steps", value=False)
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.agent = get_agent()
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def needs_research(user_input: str) -> bool:
    """Use LLM to decide if this message needs research or is just casual chat."""
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="""You are a classifier. Decide if the user's message requires internet research or is just casual conversation.
Reply with only one word: RESEARCH or CHAT.

RESEARCH: questions about facts, current events, topics, how things work, comparisons, reports, analysis.
CHAT: greetings, small talk, thank you, how are you, yes/no replies, simple acknowledgements."""),
        HumanMessage(content=user_input)
    ])
    return "RESEARCH" in response.content.upper()

if prompt := st.chat_input("Ask me anything to research..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not needs_research(prompt):
            # Casual chat — just use LLM directly, no tools
            llm = get_llm()
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else SystemMessage(content=m["content"])
                for m in st.session_state.messages[-6:]
            ]
            response = llm.invoke(history).content
            st.markdown(response)
        else:
            # Research needed — use full agent
            with st.spinner("Researching..."):
                memory = get_memory(session_id=st.session_state.session_id)
                chat_history = memory.chat_memory.messages

                result = st.session_state.agent.invoke({
                    "input": prompt,
                    "chat_history": chat_history,
                })

                response = result["output"]
                memory.chat_memory.add_user_message(prompt)
                memory.chat_memory.add_ai_message(response)

                if show_steps and result.get("intermediate_steps"):
                    with st.expander("🧠 Agent reasoning steps", expanded=False):
                        for i, (action, observation) in enumerate(result["intermediate_steps"]):
                            st.markdown(f"**Step {i+1}: `{action.tool}`**")
                            st.markdown(f"*Input:* {action.tool_input}")
                            st.markdown(f"*Observation:* {str(observation)[:500]}...")
                            st.divider()

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})