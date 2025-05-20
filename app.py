import streamlit as st
from qa import retrieve_chunks, extract_metadata_filter, answer_question
import pandas as pd

st.set_page_config(layout="wide")
st.title("Makan-AI Chatbot with Retrieval Insights")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

col1, col2 = st.columns([2, 1], gap="small")

with col1:
    st.header("Chat")
    chat_container = st.container(height=400)
    with chat_container:
        chat_scroll = st.container()
        for msg in st.session_state["messages"]:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])
        chat_scroll.markdown("</div>", unsafe_allow_html=True)
    if prompt := st.chat_input("Ask a question:"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        bot_reply = answer_question(prompt)
        st.session_state["messages"].append({"role": "bot", "content": bot_reply})
        st.rerun()

with col2:
    st.header("Retrieval Insights")
    if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "bot":
        last_user_msg = st.session_state["messages"][-2]["content"] if len(st.session_state["messages"]) >= 2 else ""
        metadata_filter = extract_metadata_filter(last_user_msg)
        retrieved_chunks = retrieve_chunks(last_user_msg, metadata_filter=metadata_filter)
        
    retrieval_container = st.container(height=400)
    with retrieval_container:
        if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "bot":
            st.markdown("**Metadata filter extracted from query:**")
            st.code(metadata_filter)
            st.markdown("**Top retrieved chunks:**")
            for chunk in retrieved_chunks[:5]:
                meta = chunk.get("metadata", {})
                st.markdown(f"- **Name:** {meta.get('name', 'N/A')} | **Region:** {meta.get('region', 'N/A')} | **Cuisine:** {meta.get('cuisine_type', 'N/A')} | **Venue:** {meta.get('venue_type', 'N/A')}")
                st.markdown(f"> {chunk['content'][:200]}...")
        else:
            st.info("Send a message to see retrieval details.")

    viz_scroll = st.container(height=400)
    with viz_scroll:
        if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "bot":
            # Prepare data for bar chart visualization
            chunk_data = []
            for i, chunk in enumerate(retrieved_chunks[:5]):
                meta = chunk.get("metadata", {})
                print(meta)
                chunk_data.append({
                    "Chunk": f"{meta.get('name', 'N/A')} ({meta.get('region', 'N/A')})",
                    "Index": i+1,
                    "Score": meta.get('similarity', meta.get('score', 1.0)),  # fallback if not present
                    "Penalised Score": meta.get('penalized_score', 0.0),
                })
            if chunk_data:
                df = pd.DataFrame(chunk_data)
                st.markdown("**Similarity Scores of Top Chunks:**")
                st.bar_chart(df.set_index("Chunk")["Score"])

# Optionally, add more advanced visualizations or download buttons here
