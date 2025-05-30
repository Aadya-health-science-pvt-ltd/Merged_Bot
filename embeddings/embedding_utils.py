# embeddings/
# Contains utilities for generating and managing vector embeddings.

# embeddings/embedding_utils.py
from typing import List, Dict
from config.llm_config import embeddings

def embed_documents_for_lancedb(docs: List[Dict], specialty: str = "paediatrics"):
    """
    Embeds a list of documents and formats them for LanceDB storage.
    Each document in 'docs' is expected to have 'text' and 'metadata'.
    """
    texts = [doc["text"] for doc in docs]
    metadatas = [{**doc["metadata"], "specialty": specialty} for doc in docs]
    
    vectors = embeddings.embed_documents(texts)
    
    data = [{
        "vector": vectors[i],
        "text": texts[i],
        "metadata": metadatas[i]
    } for i in range(len(texts))]
    
    return data

def embed_documents_from_langchain_docs(docs, specialty="paediatrics"):
    """
    Embeds Langchain Document objects and formats them for LanceDB.
    """
    texts = [doc.page_content for doc in docs]
    metadatas = [{**doc.metadata, "specialty": specialty} for doc in docs]
    
    vectors = embeddings.embed_documents(texts)
    
    data = [{
        "vector": vec,
        "text": text,
        "metadata": meta
    } for vec, text, meta in zip(vectors, texts, metadatas)]
    
    return data