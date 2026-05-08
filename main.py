import streamlit as st
import sqlite3
from app_logic import create_rag_chain
from vector_store import get_vector_store, save_vector_store
from ingest import ingest_documents
from analytics.dashboard import log_conversation
from utils.escalation import should_escalate_to_human

# ====================== DATABASE INIT ======================
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            bot_response TEXT,
            escalated BOOLEAN,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ====================== SESSION STATE ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Nexafusion HR Assistant", layout="wide")
st.title("🤖 Nexafusion HR Support")
st.caption("Elements HR Services - Ask anything about policies")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Admin Controls")
    if st.button("🔄 Re-ingest All Policies"):
        with st.spinner("Updating knowledge base..."):
            ingest_documents()
        st.success("✅ Knowledge base updated successfully!")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Recent History")
    try:
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute("SELECT user_query FROM conversations ORDER BY id DESC LIMIT 8")
        history = c.fetchall()
        conn.close()
        if history:
            for q in history:
                text = q[0][:35] + "..." if len(q[0]) > 35 else q[0]
                st.caption(f"• {text}")
        else:
            st.info("No conversations yet.")
    except:
        st.caption("History unavailable.")

# ====================== CHAT INTERFACE ======================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about leave policy, code of conduct, etc..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag_chain, retriever = create_rag_chain()
                
                # Get documents with similarity scores
                vector_store = get_vector_store()
                docs_with_score = vector_store.similarity_search_with_score(prompt, k=5)
                
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
                    response = f"{response}\n\n**Sources:**\n{sources}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Log to analytics
                log_conversation(
                    user_query=prompt,
                    bot_response=response[:500],
                    escalated=should_escalate_to_human(avg_score, prompt),
                    confidence=avg_score
                )

            except Exception as e:
                error_msg = f"❌ Error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})