import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sqlite3
import os


# Updated to ensure it finds the DB if the script is run from inside a subfolder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            user_query TEXT,
            bot_response TEXT,
            escalated BOOLEAN,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_conversation(user_query: str, bot_response: str, escalated: bool, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO conversations (timestamp, user_query, bot_response, escalated, confidence)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), user_query, bot_response[:500], escalated, confidence))
    conn.commit()
    conn.close()

def analytics_dashboard():
    st.set_page_config(page_title="Nexafusion Analytics", layout="wide")
    st.title("📊 Nexafusion HR Chatbot Analytics Dashboard")
    st.caption("Elements HR Services - Query Trends & Performance")

    init_db()

    if not os.path.exists(DB_PATH):
        st.info("No database found yet. Start chatting in the main app!")
        return

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM conversations", conn)
    conn.close()

    if df.empty:
        st.info("No conversations logged yet. Start chatting in the main app!")
        return

    # --- DATA CLEANING: FIXES THE TYPEERROR ---
    # Force confidence to be a float, turning errors into NaN (which mean() ignores)
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    # ------------------------------------------

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", len(df))
    with col2:
        st.metric("Escalations", int(df['escalated'].sum()))
    with col3:
        avg_conf = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.2f}" if not pd.isna(avg_conf) else "0.00")
    with col4:
        st.metric("Unique Queries", df['user_query'].nunique())

    # Query Trends
    st.subheader("Query Trends Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    daily = df.groupby(df['timestamp'].dt.date).size().reset_index(name='count')
    daily.columns = ['date', 'count']
    fig = px.line(daily, x='date', y='count', title="Daily Query Volume", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Top Queries
    st.subheader("Top 10 Most Asked Queries")
    top_queries = df['user_query'].value_counts().head(10)
    st.bar_chart(top_queries)

    # Escalation Rate
    st.subheader("Escalation Analysis")
    esc_rate = (df['escalated'].sum() / len(df) * 100)
    st.progress(min(esc_rate / 100, 1.0))
    st.write(f"Escalation Rate: **{esc_rate:.1f}%**")

    # Raw Data Log
    with st.expander("View Raw Conversation Logs"):
        st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True)

    if st.button("Export Full Report as CSV"):
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "nexafusion_analytics.csv", "text/csv")

if __name__ == "__main__":
    analytics_dashboard()