from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import get_vector_store
from config import GEMINI_API_KEY, LLM_TEMPERATURE

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.1,
    google_api_key=GEMINI_API_KEY
)

def create_rag_chain():
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Increased k
    
    template = """You are a helpful HR assistant for Elements HR Services.
Answer the question based on the following context from company policies.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer in a clear, professional manner:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                          for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever