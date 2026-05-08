import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import get_vector_store

# ====================== API KEY HANDLING ======================
# This works both locally (.env) and on Streamlit Cloud (secrets.toml)

if os.path.exists(".env"):
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    # For Streamlit Cloud
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ====================== LLM CONFIG ======================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",          # Better & more stable model
    temperature=0.1,
    google_api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True
)

def create_rag_chain():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}   # Increased for better context
    )
   
    template = """You are a professional and helpful HR assistant for Elements HR Services.
Answer the question based **only** on the following context from company policies.
If the answer cannot be found in the context, politely say you don't know.
Do not make up any information.

Context:
{context}

Question: {question}

Answer in a clear, professional, and concise manner:"""

    prompt = ChatPromptTemplate.from_template(template)
   
    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})\n{doc.page_content}"
            for doc in docs
        )
   
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
   
    return rag_chain, retriever
