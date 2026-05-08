import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import get_vector_store

# ====================== LOAD API KEY ======================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API Key is missing! Please add it to your .env file.")
    st.stop()

# ====================== LLM (Fixed Model Name) ======================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",   # ← Changed to this
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
)

def create_rag_chain():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
   
    template = """You are a professional HR assistant for Elements HR Services.
Answer the question using only the provided context.
If the answer is not in the context, say: "I don't know based on the provided policies."

Context:
{context}

Question: {question}

Answer professionally and clearly:"""

    prompt = ChatPromptTemplate.from_template(template)
   
    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')} "
            f"(Page {doc.metadata.get('page', '?')})\n{doc.page_content}"
            for doc in docs
        )
   
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
   
    return rag_chain, retriever
