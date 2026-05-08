import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import get_vector_store

# ====================== API KEY ======================
if os.path.exists(".env"):
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
else:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# ====================== LLM (Fixed Version) ======================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY,
    # Removed problematic parameters
)

def create_rag_chain():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
   
    template = """You are a professional HR assistant.
Use only the following context to answer the question.
If you cannot find the answer in the context, say: "I don't know based on the provided policies."

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