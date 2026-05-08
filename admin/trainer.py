import streamlit as st
from ingest import ingest_documents
from vector_store import get_vector_store, save_vector_store
import os

def admin_panel():
    st.set_page_config(page_title="Nexafusion Admin", layout="wide")
    st.title("🔧 Nexafusion Admin Panel")
    st.caption("Elements HR Services - Knowledge Base Management")

    tab1, tab2, tab3 = st.tabs(["📁 Policy Management", "🔄 Retraining", "📜 Conversation Logs"])

    with tab1:
        st.subheader("Upload New HR Policy PDFs")
        uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            os.makedirs("hr_policies", exist_ok=True)
            for file in uploaded_files:
                with open(os.path.join("hr_policies", file.name), "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"Uploaded {len(uploaded_files)} new policy document(s)")

        if st.button("🔄 Rebuild Vector Store (Hot Reload)"):
            with st.spinner("Ingesting documents and updating FAISS index..."):
                count = ingest_documents()
                st.success(f"✅ Successfully indexed {count} document chunks")

    with tab2:
        st.subheader("Neural Retrieval Settings")
        st.info("Current embedding model: sentence-transformers/all-mpnet-base-v2")
        st.info("LLM: Gemini 1.5 Pro | Temperature: 0.0")
        
        if st.button("Force Full Re-embedding"):
            with st.spinner("Clearing old index and re-embedding all documents..."):
                # Optional: delete old index and rebuild
                if os.path.exists("faiss_index"):
                    import shutil
                    shutil.rmtree("faiss_index")
                ingest_documents()
                st.success("Full re-embedding completed!")

    with tab3:
        st.subheader("Recent Conversations")
        try:
            import pandas as pd
            import sqlite3
            conn = sqlite3.connect("chat_history.db")
            logs = pd.read_sql_query("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 50", conn)
            st.dataframe(logs, use_container_width=True)
            conn.close()
        except:
            st.info("No logs available yet.")

    st.sidebar.success("Admin Panel Active - Changes apply instantly")

if __name__ == "__main__":
    admin_panel()