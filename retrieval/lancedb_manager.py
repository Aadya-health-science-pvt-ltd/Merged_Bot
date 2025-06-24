import lancedb
from langchain_community.vectorstores import LanceDB
from config.llm_config import db, embeddings
from embeddings.embedding_utils import embed_documents_for_lancedb, embed_documents_from_langchain_docs
from data.document_processor import load_and_split_doctor_info

def store_documents_once(docs, table_name):
    try:
        db.open_table(table_name)
        print(f"Table {table_name} already exists. No need to embed documents again.")
        return
    except Exception as e:
        print(f"Creating new table {table_name}...")
    
    if hasattr(docs[0], 'page_content'):
        data = embed_documents_from_langchain_docs(docs)
    else:
        data = embed_documents_for_lancedb(docs)
    
    tbl = db.create_table(table_name, data=data)
    return tbl

def setup_doctor_info_retriever():
    try:
        db.open_table("doctor_info")
        print("Doctor info table already exists. No need to embed documents again.")
        return LanceDB(connection=db, table_name="doctor_info", embedding=embeddings).as_retriever()
    except Exception as e:
        print(f"Creating new doctor info table: {e}")

    splits = load_and_split_doctor_info()
    store_documents_once(splits, "doctor_info")
    return LanceDB(connection=db, table_name="doctor_info", embedding=embeddings).as_retriever()


def rank_documents_by_relevance(docs, query, user_age_group="child"):
    ranked_docs = []
    
    for doc in docs:
        score = 0
        metadata = doc.metadata
        
        if metadata.get('is_child') == user_age_group:
            score += 50
        elif metadata.get('is_child') == 'both':
            score += 30
            
        if 'Dimensions' in metadata.get('source', ''):
            score += 40
        elif 'Clustering' in metadata.get('source', ''):
            score += 20
            
        if query.lower() in metadata.get('symptom', '').lower():
            score += 30
            
        ranked_docs.append((doc, score))
    
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_docs]

def get_retriever(table_name: str, k: int = 1):
    vector_store = LanceDB(connection=db, embedding=embeddings, table_name=table_name)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    class RankedRetriever:
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
            
        def invoke(self, query):
            docs = self.base_retriever.invoke(query)
            return rank_documents_by_relevance(docs, query)
    
    return RankedRetriever(base_retriever)