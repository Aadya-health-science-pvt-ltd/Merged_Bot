# utils/
# Contains helper utilities not specific to any other major component.

# utils/general_utils.py
import re

def needs_retrieval(query: str, classifier_chain) -> bool:
    """Determine if context retrieval is needed using LLM."""
    response = classifier_chain.invoke({"query": query}).strip().lower()
    return response == 'yes'