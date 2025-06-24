# conversation/nodes.py
from conversation.chat_state import ChatState
from models.chains import get_info_chain, symptom_chain, followup_chain, episode_check_chain
from langchain_core.messages import AIMessage
from langchain_core.documents import Document # Import for type hinting
from typing import Sequence
from retrieval.lancedb_manager import setup_doctor_info_retriever, get_retriever
from config.constants import SAMPLE_PRESCRIPTION

from utils.general_utils import needs_retrieval 
from models.chains import classifier_chain       

# Initialize retrievers (these would ideally be passed or managed in a factory)
doctor_retriever = setup_doctor_info_retriever()
retriever_dim = get_retriever("symptom_dimensions")
retriever_cls = get_retriever("symptom_classifications")

def get_info_node(state: ChatState):
    """Node to handle general information requests."""
    print("--- Executing Get Info Node ---")
    query = state["messages"][-1].content
    context_docs = doctor_retriever.invoke(query) if doctor_retriever else []
    
    response_content = get_info_chain.invoke({
        "messages": state["messages"],
        "context": context_docs
    })
    
    return {"messages": state["messages"] + [AIMessage(content=response_content)]}




def symptom_node(state: ChatState):
    print("--- Executing Symptom Node ---")
    query = state["messages"][-1].content

    # Extract metadata from state or user profile for this query
    age_group = state.get("age_group")
    gender = state.get("gender")
    specialty = state.get("specialty")

    # Build the SQL WHERE clause for LanceDB filtering
    where_clauses = []
    if age_group:
        where_clauses.append(f"is_child = '{age_group}'")
    if gender:
        where_clauses.append(f"gender = '{gender}'")
    if specialty:
        where_clauses.append(f"specialty = '{specialty}'")
    where_str = " AND ".join(where_clauses) if where_clauses else None

    combined_context = []
    if needs_retrieval(query, classifier_chain):
        print("Retrieval deemed necessary for symptom query.")
        # Use LanceDB's SQL filtering if possible
        context_dim = []
        context_cls = []
        if where_str and hasattr(retriever_dim, "vectorstore") and hasattr(retriever_dim.vectorstore, "table"):
            # Direct LanceDB table access for advanced filtering
            query_vector = retriever_dim.embed_query(query)
            context_dim = [
                Document(page_content=row["text"], metadata=row["metadata"])
                for row in retriever_dim.vectorstore.table.search(query_vector)
                    .where(where_str)
                    .limit(5)
                    .to_list()
            ]
        else:
            context_dim = retriever_dim.invoke(query) if retriever_dim else []

        if where_str and hasattr(retriever_cls, "vectorstore") and hasattr(retriever_cls.vectorstore, "table"):
            query_vector = retriever_cls.embed_query(query)
            context_cls = [
                Document(page_content=row["text"], metadata=row["metadata"])
                for row in retriever_cls.vectorstore.table.search(query_vector)
                    .where(where_str)
                    .limit(5)
                    .to_list()
            ]
        else:
            context_cls = retriever_cls.invoke(query) if retriever_cls else []

        combined_context = context_dim + context_cls
    else:
        print("Retrieval not necessary for symptom query, proceeding without extra context.")

    print("Query: ", query)
    print(f"Combined context documents: {len(combined_context)} found.")
    print(f"Context documents (metadata only): {[doc.metadata for doc in combined_context]}")
    print(f"Context documents (full): {[doc for doc in combined_context]}")

    response_content = symptom_chain.invoke({
        "messages": state["messages"],
        "context": combined_context
    })
    print("Response content: ", response_content)

    return {"messages": state["messages"] + [AIMessage(content=response_content)]}
def followup_node(state: ChatState):
    """Node to handle post-appointment follow-up."""
    print("--- Executing Follow-up Node ---")
    
    response_content = followup_chain.invoke({
        "messages": state["messages"],
        "prescription": state.get("prescription", SAMPLE_PRESCRIPTION)
    })
    
    return {"messages": state["messages"] + [AIMessage(content=response_content)]}

def same_episode_check_node(state: ChatState):
    """Node to ask the user if the current issue is the same episode."""
    print("--- Executing Same Episode Check Node ---")
    
    # Ensure configurable is accessed correctly and safe
    doctor_name_from_config = state.get("configurable", {}).get("doctor_name")
    
    previous_appointment = next((
        appt for appt in state.get("appointment_data", {}).get("appointments", []) 
        if appt.get("appt-status") == "completed" and appt.get("doctor_name") == doctor_name_from_config
    ), None)
    
    current_message = state["messages"][-1].content if state["messages"] else ""

    if previous_appointment and previous_appointment.get("symptom-summary"):
        response = episode_check_chain.invoke({
            "previous_summary": previous_appointment.get("symptom-summary"),
            "current_message": current_message
        })
        return {"messages": [AIMessage(content=f"Is this related to your previous visit for {previous_appointment.get('symptom-summary')}? Please answer 'yes' or 'no'.")], "same_episode_response": response.strip().lower()}
    else:
        return {"messages": [AIMessage(content="Let's discuss your current symptoms.")], "same_episode_response": "no"}

def process_episode_response_node(state: ChatState):
    """Node to process the user's response to the same episode question."""
    print("--- Executing Process Episode Response Node ---")
    if state.get("same_episode_response") == "yes":
        doctor_name_from_config = state.get("configurable", {}).get("doctor_name")
        previous_appointment = next((
            appt for appt in state.get("appointment_data", {}).get("appointments", []) 
            if appt.get("appt-status") == "completed" and appt.get("doctor_name") == doctor_name_from_config
        ), None)
        if previous_appointment:
            return {
                "messages": state["messages"] + [AIMessage(content="Okay, we will continue with the details from your previous visit.")],
                "prescription": previous_appointment.get("prescription"),
                "symptom_summary": previous_appointment.get("symptom-summary"),
                "patient_status": "during" # Set status to trigger symptom node
            }
        else:
            return {"messages": state["messages"] + [AIMessage(content="There was an issue retrieving your previous appointment details. Let's start with your current symptoms.")], "patient_status": "during"}
    else:
        return {"messages": state["messages"] + [AIMessage(content="Okay, let's start by discussing your current symptoms.")], "patient_status": "during"}

