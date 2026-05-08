import streamlit as st
import sqlite3
import os
from app_logic import create_rag_chain
from vector_store import get_vector_store
from utils.escalation import should_escalate_to_human
from ingest import ingest_documents
from analytics.dashboard import log_conversation

# ==========================================
# DATABASE INITIALIZATION
# ==========================================
def init_db():
    db_path = 'chat_history.db'
    
    # Connect to the DB (this creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create the table exactly as log_conversation expects it
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_query TEXT,
            bot_response TEXT,
            escalated BOOLEAN,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

# Run this the moment the app boots up
init_db()
# ==========================================

# Simplified Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Nexafusion HR Assistant", layout="wide")
st.title("🤖 Nexafusion HR Support")
st.caption("Elements HR Services - Ask anything about policies")

# ==========================================
# SIDEBAR (Now includes History!)
# ==========================================
with st.sidebar:
    st.header("Admin Controls")
    if st.button("🔄 Re-ingest All Policies"):
        with st.spinner("Updating knowledge base..."):
            ingest_documents()
        st.success("✅ Knowledge base updated!")
        
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Fetch and Display Chat History
    st.header("Recent History")
    try:
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute('''
            SELECT user_query 
            FROM conversations 
            ORDER BY id DESC 
            LIMIT 10
        ''')
        history = c.fetchall()
        conn.close()

        if history:
            for record in history:
                query = record[0]
                display_text = query if len(query) < 30 else query[:27] + "..."
                st.caption(f"🕒 {display_text}")
        else:
            st.info("No history yet. Start chatting!")
            
    except Exception as e:
        st.caption("History currently unavailable.")
# ==========================================

# Display active chat messages
for msg in st.session_state.messages if "messages" in st.session_state else []:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about leave policy, code of conduct..."):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_chain, retriever = create_rag_chain()
                
                vector_store = get_vector_store()
                docs_with_score = vector_store.similarity_search_with_score(prompt, k=3)
                
                docs = [doc for doc, score in docs_with_score]
                scores = [score for doc, score in docs_with_score]
                avg_score = sum(scores) / len(scores) if scores else 0.0

                if should_escalate_to_human(avg_score, prompt):
                    response = "🛎️ Connecting you to a live HR agent. Please wait..."
                    st.warning("Live Agent Escalation Triggered")
                else:
                    response = rag_chain.invoke(prompt)
                    sources = "\n".join([
                        f"📄 {doc.metadata.get('source', 'Unknown')} "
                        f"(Page {doc.metadata.get('page', '?')})" 
                        for doc in docs
                    ])
                    full_response = f"{response}\n\n**Sources:**\n{sources}"
                    response = full_response

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                log_conversation(
                    user_query=prompt,
                    bot_response=response[:500],
                    escalated=should_escalate_to_human(avg_score, prompt),
                    confidence=avg_score
                )

                # Force the sidebar to update immediately after logging
                st.rerun()

            except Exception as e:
                error_msg = f"❌ Error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})