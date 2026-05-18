# app.py
# Streamlit UI — entry point for the AI Research Assistant.

import streamlit as st
from core.agent import get_agent
from memory.memory import get_memory

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Assistant")
st.caption("Powered by Groq (Llama 3.3) · LangChain · Tavily")

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = get_agent()

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# ── Sidebar ────────────────────────────────────────────────────────────────────
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

# ── Chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything to research..."):
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            memory = get_memory(session_id=st.session_state.session_id)
            chat_history = memory.chat_memory.messages

            result = st.session_state.agent.invoke({
                "input": prompt,
                "chat_history": chat_history,
            })

            response = result["output"]

            # Save to memory
            memory.chat_memory.add_user_message(prompt)
            memory.chat_memory.add_ai_message(response)

            # Show reasoning steps if toggled on
            if show_steps and result.get("intermediate_steps"):
                with st.expander("🧠 Agent reasoning steps", expanded=False):
                    for i, (action, observation) in enumerate(result["intermediate_steps"]):
                        st.markdown(f"**Step {i+1}: `{action.tool}`**")
                        st.markdown(f"*Input:* {action.tool_input}")
                        st.markdown(f"*Observation:* {str(observation)[:500]}...")
                        st.divider()

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})