from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.llm_config import llm
from models.prompts import (
    get_info_prompt, followup_prompt, episode_check_prompt,
    SYMPTOM_PROMPTS, SYMPTOM_SUMMARY_PROMPT, VACCINE_SUMMARY_PROMPTS
)
from config.constants import SAMPLE_PRESCRIPTION
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re

def format_docs(docs):
    print("[DEBUG] Context passed to get_info prompt:", docs)
    if isinstance(docs, str):
        return docs
    if isinstance(docs, list):
        return "\n\n".join(docs)
    return str(docs)

get_info_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x.get("context", [])),
        clinic_name=lambda x: x.get("clinic_name", ""),
        doctor_name=lambda x: x.get("doctor_name", ""),
        services=lambda x: x.get("services", "")
    )
    | get_info_prompt
    | llm
    | StrOutputParser()
)

# Classifier system prompt
SYMPTOM_CLASSIFIER_PROMPT = '''
You are a medical triage classifier for a pediatric symptom collection bot. Given the following inputs:

- Age (in months/years): {age}
- Gender: {gender}
- Vaccine visit: {vaccine_visit} (yes/no)
- Symptom: {symptom}

Follow this logic to select the most appropriate prompt category:

1. If vaccine visit is "yes":
   - Use the nearest lower age bucket (from this list: 6w, 10w, 12w, 6m, 7m, 9m, 12m, 15m, 18m, 20m, 24m, 30m, 36m, 42m, 48m, 54m, 60m, 66m, 72m, 10y, 11y, 16y) and select the corresponding vaccine prompt.
   - If the age bucket is 10y, 11y, or 16y, output "vaccine_<age_bucket>_male" or "vaccine_<age_bucket>_female" based on the gender. For all other buckets, output "vaccine_<age_bucket>".
2. Else if age is less than 6 months:
   - Use the "less_than_6_months" prompt.
3. Else if gender is "male" and the symptom matches a male-specific symptom (e.g., testis, foreskin, penis, etc.):
   - Use the "male_child" prompt.
4. Else if gender is "female" and the symptom matches a female-specific symptom (e.g., white discharge, menstrual, breast, etc.):
   - Use the "female_child" prompt.
5. Else:
   - Use the "general_child" prompt.

Return only the category name.
Possible outputs:
- "vaccine_<age_bucket>" (e.g., "vaccine_12m", "vaccine_7m", etc.)
- "vaccine_10y_male", "vaccine_10y_female", "vaccine_11y_male", "vaccine_11y_female", "vaccine_16y_male", "vaccine_16y_female"
- "less_than_6_months"
- "male_child"
- "female_child"
- "general_child"

Do not explain your answer. Just return the category name.
'''

classifier_llm = ChatOpenAI(model="gpt-4o", temperature=0)
classifier_prompt = ChatPromptTemplate.from_template(SYMPTOM_CLASSIFIER_PROMPT)
classifier_chain = classifier_prompt | classifier_llm | StrOutputParser()

def select_symptom_prompt(age, gender, vaccine_visit, symptom):
    category = classifier_chain.invoke({
        "age": age,
        "gender": gender,
        "vaccine_visit": vaccine_visit,
        "symptom": symptom
    }).strip()
    print(f"[DEBUG] Classified prompt category: {category}")
    # Post-process for vaccine gendered keys if needed
    if category in ["vaccine_10y", "vaccine_11y", "vaccine_16y"]:
        if str(gender).lower() == "male":
            category = f"{category}_male"
        elif str(gender).lower() == "female":
            category = f"{category}_female"
    return SYMPTOM_PROMPTS.get(category, SYMPTOM_PROMPTS["general_child"])

# Dynamic symptom chain
def make_symptom_chain(age, gender, vaccine_visit, symptom, prompt_override=None):
    if prompt_override is not None:
        prompt_text = prompt_override
    else:
        prompt_text = select_symptom_prompt(age, gender, vaccine_visit, symptom)
    prompt_vars = [
        ("age", lambda x: x.get("age", "")),
        ("gender", lambda x: x.get("gender", "")),
        ("vaccine_visit", lambda x: x.get("vaccine_visit", "")),
        ("symptom", lambda x: x.get("symptom", "")),
        ("symptoms", lambda x: x.get("symptoms", "")),
        ("messages", lambda x: x.get("messages", [])),
    ]
    # Determine if this is a vaccine prompt and get the classifier category
    is_vaccine_prompt = False
    vaccine_key = None
    if prompt_override is not None:
        # Try to detect from the prompt_override string
        if 'Vaccine Visit Bot' in prompt_text or re.search(r'vaccine_\d+[mwyy]', str(prompt_text)):
            is_vaccine_prompt = True
            # Try to extract the vaccine key from the prompt_override string
            match = re.search(r'vaccine_\d+[mwyy](?:_(?:male|female))?', str(prompt_text))
            if match:
                vaccine_key = match.group(0)
    else:
        # Try to detect from the classifier category
        category = classifier_chain.invoke({
            "age": age,
            "gender": gender,
            "vaccine_visit": vaccine_visit,
            "symptom": symptom
        }).strip()
        if category.startswith("vaccine_"):
            is_vaccine_prompt = True
            vaccine_key = category
    if is_vaccine_prompt and vaccine_key in VACCINE_SUMMARY_PROMPTS:
        summary_prompt = VACCINE_SUMMARY_PROMPTS[vaccine_key]
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("system", summary_prompt),
            ("user", "{messages}")
        ])
    elif is_vaccine_prompt:
        # fallback to generic summary if key not found
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("system", SYMPTOM_SUMMARY_PROMPT),
            ("user", "{messages}")
        ])
    else:
        # For non-vaccine prompts, use only a minimal summary instruction
        minimal_summary = "At the end of the Q&A, generate a summary of the symptoms and relevant history for the doctor. Do NOT include headings like Gross Motor, Fine Motor, Speech, Social, Vision, Hearing, Feeding, Screen Exposure, Autism/ADHD/Learning Disabilities, Physical Activity, Mental Wellbeing, or Pubertal Development unless this is a vaccine visit."
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            ("system", minimal_summary),
            ("user", "{messages}")
        ])
    assign_dict = {k: v for k, v in prompt_vars}
    return RunnablePassthrough.assign(**assign_dict) | prompt | llm | StrOutputParser()

followup_chain = (
    RunnablePassthrough.assign(
        prescription=lambda x: x.get("prescription", SAMPLE_PRESCRIPTION),
        clinic_name=lambda x: x.get("clinic_name", "")
    )
    | followup_prompt
    | llm
    | StrOutputParser()
)

episode_check_chain = episode_check_prompt | llm | StrOutputParser()