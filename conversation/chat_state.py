# conversation/
# Manages conversation flow, state, and graph definitions.

# conversation/chat_state.py
from typing import TypedDict, Annotated, Sequence, Literal, Optional, List, Dict
from langchain_core.messages import BaseMessage

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
    gender: Optional[str]            # Add this
    specialty: Optional[str]         # Add this
    # --- Symptom Collector Memory/State ---
    symptom_collection_phase: Optional[str]  # "awaiting_symptom", "asking_questions", "awaiting_more_symptoms", "summary"
    collected_symptoms: Optional[List[Dict]] # List of {"symptom": str, "questions": List[str], "answers": List[str]}
    current_symptom_index: Optional[int]     # Which symptom is being processed
    current_question_index: Optional[int]    # Which question for the current symptom
