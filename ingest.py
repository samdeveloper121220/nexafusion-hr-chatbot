# import os
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from vector_store import get_vector_store
# from config import HR_POLICIES_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# def ingest_documents():
#     if not os.path.exists(HR_POLICIES_DIR):
#         os.makedirs(HR_POLICIES_DIR)
#         print(f"✅ Created folder: {HR_POLICIES_DIR}")
#         print("Please add your HR policy PDFs into this folder and run again.")
#         return 0
    
#     loader = PyPDFDirectoryLoader(HR_POLICIES_DIR)
#     docs = loader.load()
    
#     if not docs:
#         print("⚠️ No PDF files found in hr_policies folder!")
#         return 0
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ".", " ", ""]
#     )
#     splits = text_splitter.split_documents(docs)
    
#     vector_store = get_vector_store()
#     vector_store.add_documents(splits)
    
#     print(f"✅ Successfully ingested {len(splits)} chunks from {len(docs)} PDF documents")
#     return len(splits)

# if __name__ == "__main__":
#     ingest_documents()

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import get_vector_store, save_vector_store 
from config import HR_POLICIES_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def ingest_documents():
    if not os.path.exists(HR_POLICIES_DIR):
        os.makedirs(HR_POLICIES_DIR)
        print(f"✅ Created folder: {HR_POLICIES_DIR}")
        print("Please add your HR policy PDFs into this folder and run again.")
        return 0
    
    loader = PyPDFDirectoryLoader(HR_POLICIES_DIR)
    docs = loader.load()
    
    if not docs:
        print("⚠️ No PDF files found in hr_policies folder!")
        return 0
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    vector_store = get_vector_store()
    vector_store.add_documents(splits)
    
    # Save the updated memory to the hard drive
    save_vector_store(vector_store)
    
    print(f"✅ Successfully ingested {len(splits)} chunks from {len(docs)} PDF documents")
    print("💾 faiss_index has been successfully saved to disk!")
    return len(splits)

if __name__ == "__main__":
    ingest_documents()