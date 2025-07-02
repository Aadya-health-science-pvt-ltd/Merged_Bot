from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.llm_config import llm
from models.prompts import (
    get_info_prompt, symptom_prompt, followup_prompt, episode_check_prompt
)
from config.constants import SAMPLE_PRESCRIPTION

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

symptom_chain = (
    RunnablePassthrough.assign(
        clinic_name=lambda x: x.get("clinic_name", ""),
        procedure=lambda x: x.get("procedure", ""),
        doctor_name=lambda x: x.get("doctor_name", "")
    )
    | symptom_prompt
    | llm
    | StrOutputParser()
)

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