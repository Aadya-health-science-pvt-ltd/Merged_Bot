# conversation/nodes.py
from conversation.chat_state import ChatState
from models.chains import get_info_chain, followup_chain, episode_check_chain, make_symptom_chain
from langchain_core.messages import AIMessage
from typing import Dict, Any
from config.constants import SAMPLE_PRESCRIPTION
from utils.general_utils import retrieve_relevant_chunks

def get_info_node(state: ChatState):
    """Node to handle general information requests."""
    print("--- Executing Get Info Node ---")
    query = str(state["messages"][-1].content)
    # Always retrieve context, regardless of doctor_info_url
    context_chunks = retrieve_relevant_chunks(None, query, k=4)
    print(f"Retrieved {len(context_chunks)} context chunks")
    for i, chunk in enumerate(context_chunks):
        print(f"Chunk {i}: {chunk[:200]}...")
    context = "\n\n".join(context_chunks)
    print(f"Final context passed to LLM: {context[:500]}...")
    response_content = get_info_chain.invoke({
        "messages": state["messages"],
        "context": context,
        "clinic_name": state.get("clinic_name", ""),
        "doctor_name": state.get("doctor_name", ""),
        "services": state.get("services", "")
    })
    return {"messages": state["messages"] + [AIMessage(content=response_content)]}

def symptom_node(state: ChatState):
    print("--- Executing Symptom Node (Dynamic Prompts) ---")
    # Extract relevant info from state
    age = state.get("age_group", "")
    gender = state.get("gender", "")
    vaccine_visit = "yes" if "vaccine" in str(state.get("consultation_type", "")).lower() else "no"
    # Assume the first user message after bot selection is the symptom
    symptom = ""
    for msg in state["messages"]:
        if getattr(msg, "type", None) == "human" or getattr(msg, "role", None) == "user":
            symptom = getattr(msg, "content", "")
            break
    print(f"[DEBUG] Symptom Node Inputs: age={age}, gender={gender}, vaccine_visit={vaccine_visit}, symptom={symptom}")
    chain = make_symptom_chain(age, gender, vaccine_visit, symptom)
    response_content = chain.invoke({
        "age": age,
        "gender": gender,
        "vaccine_visit": vaccine_visit,
        "symptom": symptom,
        "input": symptom,
        "messages": state["messages"]
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
    
    appointment_data = state.get("appointment_data", {})
    appointments_raw = appointment_data.get("appointments", []) if appointment_data else []
    appointments = appointments_raw if isinstance(appointments_raw, list) else []
    
    # Find previous appointment with proper type handling
    previous_appointment = None
    for appt in appointments:
        if (isinstance(appt, dict) and 
            appt.get("appt-status") == "completed" and 
            appt.get("doctor_name") == doctor_name_from_config):
            previous_appointment = appt
            break
    
    current_message = str(state["messages"][-1].content) if state["messages"] else ""

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
        
        appointment_data = state.get("appointment_data", {})
        appointments_raw = appointment_data.get("appointments", []) if appointment_data else []
        appointments = appointments_raw if isinstance(appointments_raw, list) else []
        
        # Find previous appointment with proper type handling
        previous_appointment = None
        for appt in appointments:
            if (isinstance(appt, dict) and 
                appt.get("appt-status") == "completed" and 
                appt.get("doctor_name") == doctor_name_from_config):
                previous_appointment = appt
                break
                
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

