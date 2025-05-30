# lance_main.py (Revised)

# Import necessary modules from the new structure
from config.llm_config import db, llm, embeddings # Import necessary components from config
from config.constants import CLINIC_INFO, CLINIC_CONFIG, SAMPLE_PRESCRIPTION # Import constants for exposure
from data.document_processor import process_sheets # Import data processing utilities
from retrieval.lancedb_manager import setup_doctor_info_retriever, store_documents_once, get_retriever # Import LanceDB management utilities
from models.chains import chain_dim, chain_cls, classifier_chain # Import LLM chains and prompts
# Adjust import: Only import the individual graph builders, NOT build_main_graph
from conversation.graph_builder import build_get_info_graph, build_symptom_graph, build_followup_graph
from conversation.router import decide_bot_route # Import the routing logic
from conversation.chat_state import ChatState # Import ChatState for type hinting

# ====================
# Data Initialization and Storage
# ====================

# Setup Doctor Info Retriever
# This will create the 'doctor_info' table if it doesn't exist and return a retriever.
doctor_retriever = setup_doctor_info_retriever()

# Process dimension data
print("Processing symptom dimensions...")
try:
    db.open_table("symptom_dimensions")
    print("Symptom dimensions table already exists. Skipping processing.")
except Exception as e:
    print(f"Creating new symptom dimensions table: {e}")
    dim_docs = process_sheets(
        "SU.xlsx",
        chain_dim,
        sheet_filter=lambda x: "Dimensions" in x
    )
    store_documents_once(dim_docs, "symptom_dimensions")

# Process classification data
print("Processing symptom classifications...")
try:
    db.open_table("symptom_classifications")
    print("Symptom classifications table already exists. Skipping processing.")
except Exception as e:
    print(f"Creating new symptom classifications table: {e}")
    cls_docs = process_sheets(
        "SU.xlsx",
        chain_cls,
        sheet_filter=lambda x: "Clustering" in x
    )
    store_documents_once(cls_docs, "symptom_classifications")

# Initialize retrievers for conversation nodes (already done in conversation/nodes.py but ensuring it's clear)
# These lines are primarily for clarity of what's being initialized and are not strictly necessary here
# if the `conversation.nodes` module handles their initialization on import.
retriever_dim = get_retriever("symptom_dimensions")
retriever_cls = get_retriever("symptom_classifications")


# ====================
# LangGraph Setup (Individual Bots)
# ====================

# Build individual bot graphs
get_info_app = build_get_info_graph()
symptom_app = build_symptom_graph()
followup_app = build_followup_graph()

# Remove the build_main_graph() call and subsequent conditional edges setup.
# The routing is handled by app.py's send_message endpoint.

print("Individual Bot Graphs compiled.")