# utils/
# Contains helper utilities not specific to any other major component.

# utils/general_utils.py
import re

def needs_retrieval(query: str, classifier_chain) -> bool:
    """Determine if context retrieval is needed using LLM."""
    response = classifier_chain.invoke({"query": query}).strip().lower()
    return response == 'yes'


def extract_specialty_and_age(consultation_type: str):
    consultation_type = consultation_type.lower()
    if "child" in consultation_type:
        age_group = "child"
    elif "adult" in consultation_type:
        age_group = "adult"
    else:
        age_group = "both"

    if "asthma" in consultation_type or "allergy" in consultation_type:
        specialty = "allergy"
    elif "vaccination" in consultation_type:
        specialty = "vaccination"
    else:
        specialty = "general"
    return specialty, age_group