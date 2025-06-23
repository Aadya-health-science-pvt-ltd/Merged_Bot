# conversation/
# Manages conversation flow, state, and graph definitions.

# conversation/chat_state.py
from typing import TypedDict, Annotated, Sequence, Literal, Optional, List, Dict
from langchain_core.messages import BaseMessage

AppointmentData = Dict[str, any]
class ChatState(TypedDict):
    messages: List[BaseMessage]
    patient_status: Optional[Literal["pre", "during", "post"]] = None
    appointment_data: Optional[AppointmentData] = None
    prescription: Optional[str] = None
    symptom_summary: Optional[str] = None
    doctor_name: Optional[str] = None
    same_episode_response: Optional[str] = None
    age_group: Optional[str] = None         # Add this
    gender: Optional[str] = None            # Add this
    symptom: Optional[str] = None           # Add this
    specialty: Optional[str] = None         # Add this
