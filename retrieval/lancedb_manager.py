# retrieval/
# Manages vector database interactions and retrieval logic.

# retrieval/lancedb_manager.py
import lancedb
from langchain_community.vectorstores import LanceDB
from config.llm_config import db, embeddings
from embeddings.embedding_utils import embed_documents_for_lancedb, embed_documents_from_langchain_docs
from data.document_processor import load_and_split_doctor_info

def store_documents_once(docs, table_name, specialty="paediatrics"):
    """ Store documents in LanceDB with batch embedding, only if they don't exist."""
    try:
        # Check if the table exists
        db.open_table(table_name)
        print(f"Table {table_name} already exists. No need to embed documents again.")
        return
    except Exception as e:
        print(f"Creating new table {table_name}...")
    
    # Embed and prepare data
    if hasattr(docs[0], 'page_content'): # Check if it's a list of Langchain Documents
        data = embed_documents_from_langchain_docs(docs, specialty)
    else: # Assume it's a list of dicts with 'text' and 'metadata'
        data = embed_documents_for_lancedb(docs, specialty)
    
    tbl = db.create_table(table_name, data=data)
    return tbl

def setup_doctor_info_retriever(specialty="paediatrics"):
    """Process and store doctor website information, embedding only if not already present."""
    try:
        # Check if the table already exists
        db.open_table("doctor_info")
        print("Doctor info table already exists. No need to embed documents again.")
        return LanceDB(connection=db, table_name="doctor_info", embedding=embeddings).as_retriever()
    except Exception as e:
        print(f"Creating new doctor info table: {e}")

    # If the table does not exist, proceed to load and embed data
    splits = load_and_split_doctor_info()
    
    # Store the documents
    store_documents_once(splits, "doctor_info", specialty)
    return LanceDB(connection=db, table_name="doctor_info", embedding=embeddings).as_retriever()


def get_retriever(table_name: str, k: int = 2):
    """Initializes and returns a LanceDB retriever for a given table."""
    vector_store = LanceDB(connection=db, embedding=embeddings, table_name=table_name)
    return vector_store.as_retriever(search_kwargs={"k": k})