# conversation/nodes.py
from conversation.chat_state import ChatState
from models.chains import get_info_chain, symptom_chain, followup_chain, episode_check_chain
from langchain_core.messages import AIMessage
from langchain_core.documents import Document # Import for type hinting
from typing import Sequence
from retrieval.lancedb_manager import setup_doctor_info_retriever, get_retriever
from config.constants import SAMPLE_PRESCRIPTION

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
    """Node to handle symptom collection."""
    print("--- Executing Symptom Node ---")
    query = state["messages"][-1].content
    context_dim = retriever_dim.invoke(query) if retriever_dim else []
    context_cls = retriever_cls.invoke(query) if retriever_cls else []
    #@karan need help in filtering the data that is retrieved based on the following
    # Doctor speciality -- need this passed as initial data for the bot. Filter based on doctor's speciality and appointment procedure.
    # patient dimension -- child / adult and also male / femaele -- this is not being currently not being passed.
    
    combined_context = context_dim + context_cls
    
    response_content = symptom_chain.invoke({
        "messages": state["messages"],
        "context": combined_context
    })
    
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

