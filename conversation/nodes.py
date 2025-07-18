# conversation/nodes.py
from conversation.chat_state import ChatState, initialize_symptom_session
from models.chains import get_info_chain, followup_chain, episode_check_chain, make_symptom_chain
from langchain_core.messages import AIMessage
from typing import Dict, Any
from config.constants import SAMPLE_PRESCRIPTION
from utils.general_utils import retrieve_relevant_chunks
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser

def get_info_node(state: ChatState):
    """Node to handle general information requests."""
    print("--- Executing Get Info Node ---")
    query = str(state["messages"][-1].content)
    
    # Get doctor_info_url from state if available
    doctor_info_url = state.get("doctor_info_url")
    print(f"[DEBUG] doctor_info_url from state: {doctor_info_url}")
    
    # Only retrieve context if we have a valid URL
    if doctor_info_url:
        context_chunks = retrieve_relevant_chunks(doctor_info_url, query, k=4)
    print(f"Retrieved {len(context_chunks)} context chunks")
    for i, chunk in enumerate(context_chunks):
        print(f"Chunk {i}: {chunk[:200]}...")
    context = "\n\n".join(context_chunks)
    print(f"Final context passed to LLM: {context[:500]}...")
    else:
        print("[DEBUG] No doctor_info_url provided, using empty context")
        context = ""
    
    response_content = get_info_chain.invoke({
        "messages": state["messages"],
        "context": context,
        "clinic_name": state.get("clinic_name", ""),
        "doctor_name": state.get("doctor_name", ""),
        "services": state.get("services", "")
    })
    return {"messages": state["messages"] + [AIMessage(content=response_content)]}

def symptom_node(state: ChatState):
    print("--- Executing Symptom Node (Chain Externalized, Serializable State) ---")
    if "symptom_prompt" not in state or not state["symptom_prompt"]:
        raise ValueError("Hi we are initializing the Bot. Please start by saying Hi :)")
    from models.chains import make_symptom_chain
    age = state.get("age_group", "")
    gender = state.get("gender", "")
    consultation_type = state.get("consultation_type", "")
    symptom = state.get("symptoms", "")
    # Check for user stop/summary triggers
    user_message = str(state["messages"][-1].content).strip().lower() if state["messages"] else ""
    stop_triggers = ["stop", "no more", "that's all", "no more information", "done", "end", "finish", "nothing else"]
    if any(trigger in user_message for trigger in stop_triggers) or not user_message:
        # Generate summary and add 'end' flag True
        chain = make_symptom_chain(age, gender, consultation_type, symptom, prompt_override=state["symptom_prompt"])
        summary_content = chain.invoke({
            "age": age,
            "gender": gender,
            "vaccine_visit": consultation_type,
            "symptom": symptom,
            "messages": state["messages"]
        })
        return {
            "messages": state["messages"] + [AIMessage(content=summary_content)],
            "end": True
        }
    # Normal Q&A flow, end is False
    chain = make_symptom_chain(age, gender, consultation_type, symptom, prompt_override=state["symptom_prompt"])
    response_content = chain.invoke({
        "age": age,
        "gender": gender,
        "vaccine_visit": consultation_type,
        "symptom": symptom,
        "messages": state["messages"]
    })
    return {"messages": state["messages"] + [AIMessage(content=response_content)], "end": False}

def followup_node(state: ChatState):
    """Node to handle post-appointment follow-up."""
    print("--- Executing Follow-up Node ---")
    if "followup_prompt" not in state or not state["followup_prompt"]:
        raise ValueError("Hi, we are initializing the Followup Bot. Please start by saying Hi or ensure the session is initialized.")
    from models.chains import llm
    # Build a prompt template using the selected followup prompt
    followup_prompt_template = ChatPromptTemplate.from_template(state["followup_prompt"])
    # Compose the input for the chain
    input_dict = {
        "messages": state["messages"],
        "symptom_summary": state.get("symptom_summary", ""),
        "prescription": state.get("prescription", "")
    }
    response_content = (followup_prompt_template | llm | StrOutputParser()).invoke(input_dict)
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

