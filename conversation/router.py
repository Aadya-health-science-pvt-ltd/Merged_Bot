# conversation/router.py
from typing import Literal
from datetime import datetime, timezone
from langchain_core.runnables import RunnableConfig
from conversation.chat_state import ChatState

def decide_bot_route(state: ChatState, config: RunnableConfig) -> Literal["get_info", "symptom", "followup", "same_episode_check"]:
    """ Determines which bot to use based on business rules. """
    print(f"--- Routing Logic for Bot Selection ---")

    doctor_name = config.get('configurable', {}).get('doctor_name')
    appointment_data = state.get('appointment_data', {})
    
    print(f"[DEBUG] doctor_name: {doctor_name}")
    print(f"[DEBUG] appointment_data: {appointment_data}")

    current_time = datetime.now(timezone.utc)

    doctor_appointments = [
        appt for appt in appointment_data.get("appointments", [])
        if appt.get("doctor_name") == doctor_name
    ]
    
    print(f"[DEBUG] doctor_appointments: {doctor_appointments}")

    future_appointments = [
        appt for appt in doctor_appointments
        if appt.get("appt_status") == "booked" and
           (datetime.fromisoformat(appt.get("appt_datetime")).replace(tzinfo=timezone.utc) - current_time).total_seconds() < 48 * 3600
    ]
    past_appointments = [
        appt for appt in doctor_appointments
        if appt.get("appt_status") == "completed"
    ]
    
    print(f"[DEBUG] future_appointments: {future_appointments}")
    print(f"[DEBUG] past_appointments: {past_appointments}")

    first_message = state["messages"][0].content if state["messages"] else ""
    print(f"[DEBUG] first_message: '{first_message}'")

    if first_message.startswith(f"Hello {doctor_name}") and not past_appointments:
        print("Bot Router -> get_info (Rule 1)")
        return "get_info"

    if future_appointments and not past_appointments:
        print("Bot Router -> symptom (Rule 2)")
        return "symptom"

    if future_appointments and past_appointments:
        print("Bot Router -> same_episode_check (Rule 3)")
        return "same_episode_check"

    if past_appointments and not future_appointments:
        print("Bot Router -> followup (Rule 4)")
        return "followup"
    
    # Rule 5: If symptoms are provided but no appointments, route to symptom bot
    symptoms = state.get("symptoms", "")
    if symptoms and symptoms.strip():
        print("Bot Router -> symptom (Rule 5: symptoms provided)")
        return "symptom"

    print("Bot Router -> get_info (Default)")
    return "get_info"