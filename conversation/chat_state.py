# conversation/
# Manages conversation flow, state, and graph definitions.

# conversation/chat_state.py
from typing import TypedDict, Annotated, Sequence, Literal, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from utils.prompt_db import get_questioner_prompt
import re
from models.chains import classifier_chain

AppointmentData = Dict[str, any]


class ChatState(TypedDict):
    messages: List[BaseMessage]
    patient_status: Optional[Literal["pre", "during", "post"]]
    appointment_data: Optional[AppointmentData]
    prescription: Optional[str]
    symptom_summary: Optional[str]
    doctor_name: Optional[str]
    same_episode_response: Optional[str]
    age_group: Optional[str]         # Add this
    age: Optional[str]               # Add this for direct age
    gender: Optional[str]            # Add this
    vaccine_visit: Optional[str]     # Add this for vaccine visit flag
    symptoms: Optional[str]          # Renamed from 'symptom' to avoid node conflict
    specialty: Optional[str]         # Add this
    consultation_type: Optional[str]  # Add this for consultation type
    doctor_info_url: Optional[str]   # Add this for doctor's website URL
    # --- Symptom Collector Memory/State ---
    symptom_collection_phase: Optional[str]  # "awaiting_symptom", "asking_questions", "awaiting_more_symptoms", "summary"
    collected_symptoms: Optional[List[Dict]] # List of {"symptom": str, "questions": List[str], "answers": List[str]}
    current_symptom_index: Optional[int]     # Which symptom is being processed
    current_question_index: Optional[int]    # Which question for the current symptom
    symptom_prompt: Optional[str]            # Stores the selected prompt for the session

def initialize_symptom_session(state: ChatState):
    """Initializes the symptom session by running the classifier and storing the selected prompt in state."""
    age = state.get("age", "") or state.get("age_group", "")
    gender = state.get("gender", "")
    consultation_type = state.get("consultation_type", "")
    vaccine_visit = state.get("vaccine_visit", "")
    symptom = state.get("symptoms", "")

    # Fallback: If consultation_type contains 'vaccine', set vaccine_visit to 'yes'
    if not vaccine_visit and consultation_type and "vaccine" in consultation_type.lower():
        vaccine_visit = "yes"
        state["vaccine_visit"] = "yes"

    # Fallback: If age is not a number, try to extract from other fields or set to a default
    if not age or not re.search(r"\d+", str(age)):
        print(f"[WARNING] Invalid or missing age: '{age}'. Defaulting to '9 months'.")
        age = "9 months"
        state["age"] = "9 months"

    # Fallback: If gender is missing, log a warning and set to 'unknown'
    if not gender:
        print(f"[WARNING] Gender not provided in state. Defaulting to 'unknown'.")
        gender = "unknown"
        state["gender"] = "unknown"

    # Improved vaccine visit detection
    consultation_type = state.get("consultation_type", "")
    if not vaccine_visit and consultation_type:
        vaccine_keywords = ["vaccine", "vaccination", "immunization", "shot"]
        if any(keyword in consultation_type.lower() for keyword in vaccine_keywords):
            vaccine_visit = "yes"
            state["vaccine_visit"] = "yes"
    if not vaccine_visit and symptom:
        vaccine_keywords = ["vaccine", "vaccination", "immunization", "shot"]
        if any(keyword in symptom.lower() for keyword in vaccine_keywords):
            vaccine_visit = "yes"
            state["vaccine_visit"] = "yes"

    classifier_input = {
        "age": age,
        "gender": gender,
        "vaccine_visit": vaccine_visit,
        "consultation_type": consultation_type,
        "symptom": symptom
    }
    print("[DEBUG] Classifier input:", classifier_input)
    classifier_output = classifier_chain.invoke(classifier_input).strip()
    print("[DEBUG] Classifier output:", classifier_output)

    # Always fetch prompt dynamically
    selected_prompt = get_questioner_prompt(classifier_output)
    if not selected_prompt:
        print(f"[WARNING] Could not fetch prompt for key '{classifier_output}'. Trying vaccine/age fallback.")
        age_str = str(age).lower().strip()
        month_match = re.match(r"([\d.]+)\s*(m|mo|mos|month|months)", age_str)
        year_match = re.match(r"([\d.]+)\s*(y|yr|yrs|year|years)", age_str)
        months = None
        if month_match:
            try:
                months = float(month_match.group(1))
            except Exception:
                months = None
        elif year_match:
            try:
                years = float(year_match.group(1))
                months = years * 12
            except Exception:
                months = None
        else:
            try:
                val = float(re.findall(r"[\d.]+", age_str)[0])
                if val < 6:
                    months = val
                else:
                    months = val * 12
            except Exception:
                months = None
        fallback_key = "less_than_6_months" if months is not None and months < 6 else "general_child"
        selected_prompt = get_questioner_prompt(fallback_key)
        if not selected_prompt:
            print(f"[ERROR] Could not fetch fallback prompt for key '{fallback_key}'. Using default message.")
            selected_prompt = "I'm sorry, I couldn't load the right questions. Please try again later."
    print("[DEBUG] Final selected prompt:\n", selected_prompt)
    state["symptom_prompt"] = selected_prompt
    return state
