import os
from typing import List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader

# Updated imports for LangChain v0.2+
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

def read_pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        texts.append(f"\n\n--- Page {i+1} ---\n{page_text}")
    return "\n".join(texts)

def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[Document]:
    # RecursiveCharacterTextSplitter keeps semantically related text together
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]

def build_vectorstore_from_pdf(pdf_path: str) -> FAISS:
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY. Put it in your .env file.")

    text = read_pdf_to_text(pdf_path)
    docs = chunk_text(text)

    # Local embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vs = FAISS.from_documents(docs, embeddings)
    return vs

def answer_question(vectorstore, question: str, chat_history: list, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Updated to use .invoke() instead of get_relevant_documents
    context_docs = retriever.invoke(question)
    
    if not context_docs:
        return "I couldn't find any relevant information in the uploaded PDF."

    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Format chat history for context
    history_text = ""
    for u, a in chat_history[-6:]:
        history_text += f"User: {u}\nAssistant: {a}\n"

    system_prompt = (
        "You are a helpful assistant answering strictly from the PDF context.\n"
        "If the answer is not in the context, say: 'I don't know based on the PDF.'\n"
        "Keep answers concise."
    )

    final_prompt = f"PDF CONTEXT:\n{context}\n\nCHAT HISTORY:\n{history_text}\n\nQUESTION:\n{question}"

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    # LLM call also uses .invoke()
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ])

    return response.content