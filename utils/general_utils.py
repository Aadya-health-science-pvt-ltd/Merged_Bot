# utils/
# Contains helper utilities not specific to any other major component.

# utils/general_utils.py
import re
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
import tiktoken
from langchain_openai import OpenAIEmbeddings
import os
import hashlib
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

def needs_retrieval(query: str, classifier_chain) -> bool:
    """Determine if context retrieval is needed using LLM."""
    response = classifier_chain.invoke({"query": query}).strip().lower()
    return response == 'yes'


def extract_specialty_and_age(consultation_type: str):
    consultation_type = consultation_type.lower()
    if "child" in consultation_type:
        age_group = "child"
    elif "adult" in consultation_type:
        age_group = "adult"
    else:
        age_group = "both"

    if "asthma" in consultation_type or "allergy" in consultation_type:
        specialty = "allergy"
    elif "vaccination" in consultation_type:
        specialty = "vaccination"
    else:
        specialty = "general"
    return specialty, age_group

# --- Web Scraping ---
def scrape_and_clean_text(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text = "\n".join(doc.page_content for doc in docs)
    return text

# --- Chunking ---
def chunk_text(text, max_tokens=350, overlap=50):
    enc = tiktoken.get_encoding('cl100k_base')
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# --- Embedding ---
embeddings = OpenAIEmbeddings()

def embed_chunks(chunks):
    return embeddings.embed_documents(chunks)

# --- FAISS DB Management ---
def get_faiss_db_path(url=None):
    return "faiss_main"

def build_or_load_faiss(url, force_rebuild=False):
    db_path = get_faiss_db_path()
    print(f"[DEBUG] build_or_load_faiss: db_path={db_path}, force_rebuild={force_rebuild}")
    
    # If no URL is provided, return None to indicate no context available
    if not url:
        print("[DEBUG] No URL provided, returning None")
        return None
        
    if not force_rebuild and os.path.exists(db_path):
        print(f"Loading FAISS DB from {db_path}")
        return FAISS.load_local(db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print(f"Scraping and building FAISS DB for {url}")
    text = scrape_and_clean_text(url)
    print(f"Scraped text length: {len(text)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    print(f"Number of chunks: {len(docs)}")
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local(db_path)
    print(f"FAISS DB saved to {db_path}")
    return db

def retrieve_relevant_chunks(url, query, k=4):
    print(f"[DEBUG] Entered retrieve_relevant_chunks with query: '{query}'")
    db = build_or_load_faiss(url)  # Use the provided URL
    if db is None:
        print("[DEBUG] No FAISS DB available, returning empty list")
        return []
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    print(f"[DEBUG] Retrieved {len(docs)} docs from retriever for query: '{query}'")
    for i, doc in enumerate(docs):
        print(f"[DEBUG] Chunk {i}: {doc.page_content[:200]}...")
    return [doc.page_content for doc in docs]