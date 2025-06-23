from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config.llm_config import llm
from models.prompts import (
    DIM_PROMPT, CLS_PROMPT, CLASSIFIER_PROMPT, 
    get_info_prompt, symptom_prompt, followup_prompt, episode_check_prompt
)
from config.constants import CLINIC_INFO, CLINIC_CONFIG, SAMPLE_PRESCRIPTION
from typing import Sequence
from langchain_core.documents import Document

def format_docs(docs: Sequence[Document]) -> str:
    if not docs:
        return "No context retrieved."
    return "\n\n".join(doc.page_content for doc in docs)

def format_prioritized_context(docs: Sequence[Document]) -> str:
    if not docs:
        return "No specific context available for this query."
    
    primary_doc = docs[0]
    context = f"PRIMARY CONTEXT (highest priority):\n{primary_doc.page_content}"
    
    if len(docs) > 1:
        additional_context = "\n\n".join([doc.page_content for doc in docs[1:3]])
        context += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"
    
    return context

classifier_chain = CLASSIFIER_PROMPT | llm | StrOutputParser()
chain_dim = DIM_PROMPT | llm | StrOutputParser()
chain_cls = CLS_PROMPT | llm | StrOutputParser()

get_info_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"]),
        clinic_name=lambda _: CLINIC_INFO["name"],
        doctor_name=lambda _: CLINIC_INFO["doctor"],
        services=lambda _: CLINIC_INFO["services"]
    )
    | get_info_prompt
    | llm
    | StrOutputParser()
)

symptom_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_prioritized_context(x["context"]),
        clinic_name=lambda _: CLINIC_CONFIG["clinic_name"],
        procedure=lambda _: CLINIC_CONFIG["procedure"],
        doctor_name=lambda _: CLINIC_CONFIG["doctor_name"]
    )
    | symptom_prompt
    | llm
    | StrOutputParser()
)

followup_chain = (
    RunnablePassthrough.assign(
        prescription=lambda x: x.get("prescription", SAMPLE_PRESCRIPTION),
        clinic_name=lambda _: CLINIC_INFO["name"]
    )
    | followup_prompt
    | llm
    | StrOutputParser()
)

episode_check_chain = episode_check_prompt | llm | StrOutputParser()