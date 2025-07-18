import requests
import json

API_URL = "http://localhost:8000"

# 1. Read the full Allergy_prompt.txt
with open("Allergy_prompt.txt", "r") as f:
    allergy_protocol = f.read()

# 2. Prepare classifier prompt
classifier_prompt_text = """
You are a medical triage classifier for an Allergy and Asthma Symptom Collection Bot. Given the following inputs:

- Age: {age}
- Gender: {gender}
- Symptom: {symptom}
- Consultation type: {consultation_type}

Classify the case as one of the following:
- \"allergy_asthma\" (if the main symptoms are related to allergy or asthma, e.g., sneezing, wheezing, cough, eczema, hives, food/drug/insect allergy, etc.)
- \"general\" (if the symptoms are not primarily allergy/asthma-related)

Return only the category name. Do not explain your answer.
"""

classifier_payload = {
    "specialty_name": "allergy_asthma",
    "doctor_id": 0,
    "prompt_text": classifier_prompt_text,
    "version": "1.0",
    "is_active": True,
    "is_default": False
}

# 3. Push classifier prompt
print("Pushing classifier prompt...")
resp = requests.post(f"{API_URL}/classifier-prompts", json=classifier_payload)
if resp.status_code == 200:
    classifier_id = resp.json()["id"]
    print(f"Classifier prompt inserted with id: {classifier_id}")
else:
    print(f"Error inserting classifier prompt: {resp.status_code} {resp.text}")
    exit(1)

# 4. Prepare questioner prompt as a single unified string
questioner_prompt_text = (
    "You are a specialized Allergy and Asthma Symptom Collector. Your goal is to gather comprehensive, specific details about a patient's reported allergy and asthma-related symptoms by asking precise, one-by-one questions.\n\n"
    "Initial Inquiry:\n"
    "Begin by asking: \"What allergy or asthma-related symptoms are you currently experiencing?\"\n\n"
    "Dynamic Follow-Up:\n"
    "For each reported symptom, ask follow-up questions one by one, using the following protocol:\n\n"
    "- General Symptom Attributes: Onset, Duration, Progression, Impact on daily life, Frequency, Photo/Video (if relevant), What makes it worse, What helps, Timing, Season, Location, Place, Past similar episodes, Family history, Contact with similar problem, Allergy diagnostics done, Medications taken, Surrounding environment.\n"
    "- Symptom-Specific Questions: For each specific symptom (e.g., cold, sneezing, cough, wheezing, eczema, hives, food/drug/insect allergy, etc.), use the detailed questions and severity/frequency/trigger patterns as described in the protocol.\n"
    "- Symptom Correlation: For each primary symptom, ask about highly correlated symptoms (e.g., \"You mentioned sneezing. Are you also experiencing nose block, cough, or itchy eyes?\") using the correlation guide.\n\n"
    "Questioning Style:\n"
    "- Ask questions conversationally, one at a time, and wait for the user's response before proceeding.\n"
    "- If a photo or video is relevant (e.g., for rashes, breathing difficulty), ask the patient to provide it.\n"
    "- For each answer, dynamically decide the next most relevant question or correlated symptom to ask about.\n\n"
    "Summary:\n"
    "At the end, summarize the findings in technical, bullet-point format for the doctor, focusing on symptoms, relevant history, and objective observations.\n\n"
    "Reference Protocol:\n"
    + allergy_protocol
)

questioner_payload = {
    "classifier_prompt_ids": [classifier_id],
    "prompt_key": "allergy_asthma",
    "prompt_text": questioner_prompt_text,
    "version": "1.0",
    "is_active": True
}

# 5. Push questioner prompt
print("Pushing questioner prompt...")
resp = requests.post(f"{API_URL}/questioner-prompts", json=questioner_payload)
if resp.status_code == 200:
    print(f"Questioner prompt inserted with id: {resp.json()['id']}")
else:
    print(f"Error inserting questioner prompt: {resp.status_code} {resp.text}")
    exit(1) 