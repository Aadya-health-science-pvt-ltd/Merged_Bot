# conversation/
# Manages conversation flow, state, and graph definitions.

# conversation/chat_state.py
from typing import TypedDict, Annotated, Sequence, Literal, Optional, List, Dict, Any
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
    age: Optional[str]               # Add this for direct age
    gender: Optional[str]            # Add this
    vaccine_visit: Optional[str]     # Add this for vaccine visit flag
    symptoms: Optional[str]          # Renamed from 'symptom' to avoid node conflict
    specialty: Optional[str]         # Add this
    doctor_info_url: Optional[str]   # Add this for doctor's website URL
    # --- Symptom Collector Memory/State ---
    symptom_collection_phase: Optional[str]  # "awaiting_symptom", "asking_questions", "awaiting_more_symptoms", "summary"
    collected_symptoms: Optional[List[Dict]] # List of {"symptom": str, "questions": List[str], "answers": List[str]}
    current_symptom_index: Optional[int]     # Which symptom is being processed
    current_question_index: Optional[int]    # Which question for the current symptom
    symptom_prompt: Optional[str]            # Stores the selected prompt for the session

def initialize_symptom_session(state: ChatState):
    """Initializes the symptom session by running the classifier and storing the selected prompt in state."""
    from models.chains import classifier_chain, SYMPTOM_PROMPTS
    import re
    # Prefer 'age' over 'age_group', and ensure it's a string like '9 months' or '2 years'
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

    classifier_input = {
        "age": age,
        "gender": gender,
        "vaccine_visit": vaccine_visit,
        "symptom": symptom
    }
    print("[DEBUG] Classifier input:", classifier_input)
    classifier_output = classifier_chain.invoke(classifier_input).strip()
    print("[DEBUG] Classifier output:", classifier_output)
    # Fallback logic if classifier output is not a valid prompt key
    selected_prompt = None
    fallback_reason = None
    if classifier_output not in SYMPTOM_PROMPTS:
        print(f"[WARNING] Classifier output '{classifier_output}' not found in SYMPTOM_PROMPTS. Attempting to select closest vaccine prompt.")
        age_str = str(age).lower().strip()
        # Accept more suffixes and decimals
        month_match = re.match(r"([\d.]+)\s*(m|mo|mos|month|months)", age_str)
        year_match = re.match(r"([\d.]+)\s*(y|yr|yrs|year|years)", age_str)
        months = None
        years = None
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
                years = None
                months = None
        else:
            # Try to parse as just a number (assume months if <6, years if >6)
            try:
                val = float(re.findall(r"[\d.]+", age_str)[0])
                if val < 6:
                    months = val
                else:
                    years = val
                    months = years * 12
            except Exception:
                months = None
                years = None
        # Vaccine prompt fallback
        if vaccine_visit and vaccine_visit.lower() == "yes":
            vaccine_month_buckets = sorted([
                int(k.split('_')[1][:-1]) for k in SYMPTOM_PROMPTS.keys()
                if k.startswith('vaccine_') and k.endswith('m') and k.split('_')[1][:-1].isdigit()
            ])
            vaccine_year_buckets = sorted([
                int(k.split('_')[1][:-1]) for k in SYMPTOM_PROMPTS.keys()
                if k.startswith('vaccine_') and k.endswith('y') and k.split('_')[1][:-1].isdigit()
            ])
            if months is not None:
                exact_key = f"vaccine_{int(months)}m"
                if exact_key in SYMPTOM_PROMPTS:
                    print(f"[DEBUG] Exact match for age bucket: {exact_key}")
                    selected_prompt = SYMPTOM_PROMPTS[exact_key]
                    fallback_reason = f"Exact vaccine month match: {exact_key}"
                else:
                    lower_buckets = [b for b in vaccine_month_buckets if b < months]
                    if lower_buckets:
                        closest = max(lower_buckets)
                        age_bucket = f"vaccine_{closest}m"
                        print(f"[DEBUG] Fallback to lower age bucket: {age_bucket}")
                        selected_prompt = SYMPTOM_PROMPTS[age_bucket]
                        fallback_reason = f"Lower vaccine month bucket: {age_bucket}"
                    else:
                        print("[WARNING] Could not find a matching vaccine prompt for age. Using less_than_6_months if <6m, else general_child.")
                        if months is not None and months < 6:
                            selected_prompt = SYMPTOM_PROMPTS["less_than_6_months"]
                            fallback_reason = "No vaccine prompt, <6m: less_than_6_months"
                        else:
                            selected_prompt = SYMPTOM_PROMPTS["general_child"]
                            fallback_reason = "No vaccine prompt, >=6m: general_child"
            elif years is not None:
                exact_key = f"vaccine_{int(years)}y"
                if exact_key in SYMPTOM_PROMPTS:
                    print(f"[DEBUG] Exact match for age bucket: {exact_key}")
                    selected_prompt = SYMPTOM_PROMPTS[exact_key]
                    fallback_reason = f"Exact vaccine year match: {exact_key}"
                else:
                    lower_buckets = [b for b in vaccine_year_buckets if b < years]
                    if lower_buckets:
                        closest = max(lower_buckets)
                        age_bucket = f"vaccine_{closest}y"
                        print(f"[DEBUG] Fallback to lower year bucket: {age_bucket}")
                        selected_prompt = SYMPTOM_PROMPTS[age_bucket]
                        fallback_reason = f"Lower vaccine year bucket: {age_bucket}"
                    else:
                        print("[WARNING] Could not find a matching vaccine prompt for age. Using less_than_6_months if <6m, else general_child.")
                        if years is not None and years < 0.5:
                            selected_prompt = SYMPTOM_PROMPTS["less_than_6_months"]
                            fallback_reason = "No vaccine prompt, <6m: less_than_6_months"
                        else:
                            selected_prompt = SYMPTOM_PROMPTS["general_child"]
                            fallback_reason = "No vaccine prompt, >=6m: general_child"
            else:
                print("[WARNING] Could not parse age for vaccine prompt. Using less_than_6_months if <6m, else general_child.")
                # If age is blank or not parseable, default to less_than_6_months
                if not age or (months is not None and months < 6):
                    selected_prompt = SYMPTOM_PROMPTS["less_than_6_months"]
                    fallback_reason = "Unparseable age, fallback: less_than_6_months"
                else:
                    selected_prompt = SYMPTOM_PROMPTS["general_child"]
                    fallback_reason = "Unparseable age, fallback: general_child"
        else:
            # Not a vaccine visit, fallback by age
            if months is not None and months < 6:
                selected_prompt = SYMPTOM_PROMPTS["less_than_6_months"]
                fallback_reason = "Non-vaccine, <6m: less_than_6_months"
            else:
                selected_prompt = SYMPTOM_PROMPTS["general_child"]
                fallback_reason = "Non-vaccine, >=6m or unknown: general_child"
        print(f"[DEBUG] Fallback selected prompt: {fallback_reason}")
    else:
        selected_prompt = SYMPTOM_PROMPTS[classifier_output]
        print(f"[DEBUG] Classified prompt key: {classifier_output}")
    print("[DEBUG] Classified prompt text:\n", selected_prompt)
    state["symptom_prompt"] = selected_prompt
    return state
