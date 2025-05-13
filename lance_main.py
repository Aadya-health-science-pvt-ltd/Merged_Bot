
# ---- Cell 0 ----
import pandas as pd
import lancedb
from datetime import datetime, timezone
import pyarrow as pa # Import pyarrow for schema definition if needed
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import langchain_core
import re # Import regex for clarification check
import os # Import os for file existence check
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document


# ---- Cell 1 ----
load_dotenv()

# ====================
# Configuration
# ====================
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"
DOCTOR_WEBSITE_URL = "https://www.linqmd.com/doctor/p-v-n-sravanthi"
CLINIC_INFO = {
    "name": "Metro Allergy Clinic",
    "doctor": "Dr. Emily Johnson", # Example, will be overridden by CLINIC_CONFIG if used by that chain
    "services": "Allergy Testing, Immunotherapy, Pediatric Allergies",
    "welcome": "Hello! I am the Metro Allergy Clinic Assistant. How can I help you today?" # Generic welcome if needed
}
CLINIC_CONFIG = { # Specific config for symptom bot
    "clinic_name": "Allergy Central",
    "procedure": "Allergy",
    "doctor_name": "Dr Balachandra BV"
}
SAMPLE_PRESCRIPTION = """Prescribed Medications:
- Loratadine 10mg: 1 tablet daily AM
- Epinephrine auto-injector: 0.3mg IM PRN
- Follow-up in 2 weeks"""

# Initialize core components
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=BATCH_SIZE)
db = lancedb.connect("./lancedb")


def get_appointment_data(patient_id: str) -> dict:
    """Fetches appointment data for a patient. Replace this with your actual database logic."""
    # Replace this with your actual appointment data fetching logic
    return {
        "patient-id": patient_id,
        "patient-name": "Karan",
        "patient-mobile": "99999999",
        "appointments": [
            {
                "appt-id": "1",
                "appt-datetime": "2025-04-21T18:25:00-05:30",
                "appt-status": "completed",
                "doctor-name": "Dr. Balachandra BV",
                "procedure-name": "Allergy Consultation",
                "symptom-summary": "Sneezing",
                "prescrption": "Sample prescription"
            }
        ]
    }


# ---- Cell 2 ----
# ====================
# Doctor Info RAG Setup
# ====================
def setup_doctor_info_retriever(specialty="paediatrics"):
    """Process and store doctor website information, embedding only if not already present."""
    try:
        # Check if the table already exists
        db.open_table("doctor_info")
        print("Doctor info table already exists. No need to embed documents again.")
        return LanceDB(connection=db, table_name="doctor_info", embedding=embeddings).as_retriever()
    except Exception as e:
        print(f"Creating new doctor info table: {e}")

    # If the table does not exist, proceed to load and embed data
    loader = WebBaseLoader(DOCTOR_WEBSITE_URL)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(docs)
    
    texts = [doc.page_content for doc in splits]
    metadatas = [{"source": DOCTOR_WEBSITE_URL, "specialty": specialty} for _ in texts]
    vectors = embeddings.embed_documents(texts)
    
    data = [{
        "vector": vec,
        "text": text,
        "metadata": meta
    } for vec, text, meta in zip(vectors, texts, metadatas)]
    
    # Create the table with the embedded data
    tbl = db.create_table("doctor_info", data=data)
    return LanceDB(connection=db, table_name="doctor_info", embedding=embeddings).as_retriever()


doctor_retriever = setup_doctor_info_retriever()



# ---- Cell 5 ----
# Define prompt templates
DIM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Generate question sequence for symptom assessment. Prioritize by scores (100=highest):"),
    ("human", "{row_data}")
])

CLS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Identify related symptoms based on specialist priorities (100=most relevant):"),
    ("human", "{row_data}")
])

# Context classifier components
CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""
Analyze if this patient input requires medical context retrieval. 
Consider these categories as needing retrieval:
- Symptom descriptions
- Medical history details
- Specific health concerns
- Medication questions

Respond with ONLY 'yes' or 'no'.

Input: {query}
Requires medical context?""")

classifier_chain = CLASSIFIER_PROMPT | llm | StrOutputParser()

def needs_retrieval(query: str) -> bool:
    """Determine if context retrieval is needed using LLM"""
    response = classifier_chain.invoke({"query": query}).strip().lower()
    return response == 'yes'

# Create chains
chain_dim = DIM_PROMPT | llm | StrOutputParser()
chain_cls = CLS_PROMPT | llm | StrOutputParser()


# ---- Cell 6 ----
def process_batch(df, chain, sheet_name, specialty="paediatrics"):
    """Process a batch of rows with error handling"""
    results = []
    for _, row in df.iterrows():
        try:
            row_dict = row.where(pd.notna(row), None).to_dict()
            description = chain.invoke({"row_data": str(row_dict)})
            results.append({
                "text": description,
                "metadata": {
                    "symptom": str(row_dict.get("Symptoms", "Unknown")).strip(),
                    "is_child": "child" if "child" in sheet_name.lower() else "both",
                    "gender": "female" if row_dict.get("Only Female", 0) else "both",
                    "source": sheet_name,
                    "specialty": specialty
                }
            })
        except Exception as e:
            print(f"Error processing row: {e}")
    return results

def process_sheets(file_path, chain, sheet_filter):
    """Process all sheets with batch parallel processing"""
    xls = pd.ExcelFile(file_path)
    all_docs = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for sheet_name in xls.sheet_names:
            if not sheet_filter(sheet_name):
                continue
                
            df = xls.parse(sheet_name)
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i:i+BATCH_SIZE]
                futures.append(
                    executor.submit(process_batch, batch, chain, sheet_name)
                )
        
        for future in futures:
            all_docs.extend(future.result())
    
    return all_docs

# ---- Cell 6.9 ----
def store_documents_once(docs, table_name, specialty="paediatrics"):
    """ Store documents in LanceDB with batch embedding, only if they don't exist."""
    try:
        # Check if the table exists
        db.open_table(table_name)
        print(f"Table {table_name} already exists. No need to embed documents again.")
        return
    except Exception as e:
        print(f"Creating new table {table_name}...")
    
    # Add specialty to metadata
    texts = [doc["text"] for doc in docs]
    metadatas = [{**doc["metadata"], "specialty": specialty} for doc in docs]
    
    vectors = embeddings.embed_documents(texts)
    
    data = [{
        "vector": vectors[i],
        "text": texts[i],
        "metadata": metadatas[i]
    } for i in range(len(texts))]
    
    tbl = db.create_table(table_name, data=data)
    return tbl


# ---- Cell 7 ----
# Process dimension data
print("Processing symptom dimensions...")
try:
    db.open_table("symptom_dimensions")
    print("Symptom dimensions table already exists. Skipping processing.")
    dim_docs = []  # Initialize as empty list if not processing
except Exception as e:
    print(f"Creating new symptom dimensions table: {e}")
    dim_docs = process_sheets(
        "SU.xlsx",
        chain_dim,
        sheet_filter=lambda x: "Dimensions" in x
    )
    store_documents_once(dim_docs, "symptom_dimensions")

# ---- Cell 8 ----
# Process classification data
print("Processing symptom classifications...")
try:
    db.open_table("symptom_classifications")
    print("Symptom classifications table already exists. Skipping processing.")
    cls_docs = []  # Initialize as empty list if not processing
except Exception as e:
    print(f"Creating new symptom classifications table: {e}")
    cls_docs = process_sheets(
        "SU.xlsx",
        chain_cls,
        sheet_filter=lambda x: "Clustering" in x
    )
    store_documents_once(cls_docs, "symptom_classifications")



# ---- Cell 12 ----
# Initialize retrievers
vector_store_dim = LanceDB(connection=db, embedding=embeddings, table_name="symptom_dimensions")
vector_store_cls = LanceDB(connection=db, embedding=embeddings, table_name="symptom_classifications")

retriever_dim = vector_store_dim.as_retriever(search_kwargs={"k": 2})
retriever_cls = vector_store_cls.as_retriever(search_kwargs={"k": 2})



# ---- Cell 15 ----


# ====================
# State Definition
# ====================
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Optional[Sequence[Document]] # Store retrieved context if needed later
    patient_status: Optional[Literal["pre", "during", "post"]] # 'pre', 'during', 'post' or None if unknown
    prescription: Optional[str] # Store prescription for follow-up
    needs_clarification: bool # Flag to indicate if clarification was just asked


# ---- Cell 16 ----
# ====================
# Bot Chains
# ====================

# 1. Get-Info Bot (RAG)
get_info_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a clinic information assistant for {clinic_name}.
Use the following retrieved context about the clinic and doctor to answer questions:
{context}

Clinic Details (Fallback if context is missing):
- Name: {clinic_name}
- Doctor: {doctor_name}
- Services: {services}

Conversation History:
{history}

Rules:
1. Answer questions concisely based *primarily* on the retrieved context if available.
2. Use the fallback details only if the context doesn't provide the answer.
3. If unsure or asked for medical advice, politely state you cannot answer and offer to connect to human staff.
4. Base your answer on the current question, considering the history for context.

Current Question: {input}
Answer:"""),
    # MessagesPlaceholder(variable_name="messages") # History is now passed explicitly
])

def format_docs(docs: Sequence[Document]) -> str:
    """Helper function to format retrieved documents."""
    if not docs:
        return "No context retrieved."
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(msgs: Sequence[BaseMessage]) -> str:
    """Helper function to format message history."""
    return "\n".join(f"{msg.type.upper()}: {msg.content}" for msg in msgs)


get_info_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"]),
        history=lambda x: format_history(x["messages"][:-1]), # Pass history excluding current input
        input=lambda x: x["messages"][-1].content, # Isolate current input
        # Provide fallback values from CLINIC_INFO
        clinic_name=lambda _: CLINIC_INFO["name"],
        doctor_name=lambda _: CLINIC_INFO["doctor"],
        services=lambda _: CLINIC_INFO["services"]
    )
    | get_info_prompt
    | llm
    | StrOutputParser()
)


# 2. Symptom Collector Bot (MODIFIED PROMPT)
SYMPTOM_SYSTEM_PROMPT = """You are a secretary bot named Genie at {clinic_name}. Your goal is to gather allergy symptoms by asking **one question at a time**.

**CRITICAL INSTRUCTIONS:**
1.  **Review the Conversation History CAREFULLY** before deciding what to ask next.
2.  **DO NOT REPEAT QUESTIONS** that have already been asked or answered in the history.
3.  Ask the **NEXT LOGICAL QUESTION** based on the user's previous answers and the typical flow of symptom gathering.
4.  Use the Retrieved Context below *only* as a general guide for the *types* of questions relevant to the mentioned symptoms, but **prioritize the Conversation History** to determine the *specific* question to ask next.

**Conversation Flow:**
* **First Interaction:** If the history shows you haven't started collecting symptoms yet, begin with: "Okay, let's discuss your symptoms for {doctor_name}. Can you start by telling me what {procedure} symptoms bring you in today?"
* **Subsequent Interactions:** Based on the **Conversation History**, identify the last question asked and the user's answer. Ask the **next relevant question** from the standard symptom details list below.

**Standard Symptom Details to Ask (One at a time, in order, checking history first):**
    1.  Primary symptom(s).
    2.  Onset (How did it start? Sudden, gradual, etc.)
    3.  Duration (How long have you had it? Days, weeks, months, years?)
    4.  Progression (Improved, worsened, stayed the same?)
    5.  Severity (Mild, moderate, severe? Scale 1-10?)
    6.  Frequency (How often? Constant, intermittent?) - If applicable.
    7.  Impact on lifestyle (Sleep, work, school?)
    8.  Triggers/What makes it worse? (Specific foods, dust, pollen, activities, etc.)
    9.  What makes it better? (Medications tried, remedies?)
    10. Timing/Seasonality (Worse at certain times of day or year?) - If relevant (e.g., not for food allergies).
    11. Location (Where on the body?) - Ask only if relevant (e.g., for rashes, swelling; NOT for cough/sneeze).
    12. Associated symptoms (Other symptoms occurring at the same time?)
    13. Family history (Similar issues in the family?)

**Other Important Behaviors:**
* Always verify unclear responses.
* If user says "stop", "exit", "quit", or similar: Generate a summary based on the history.
* Only discuss {procedure}-related topics. Acknowledge non-{procedure} symptoms and state they can be discussed with the doctor.
* Do not provide remedies or prescriptions.
* Never ask more than one question at a time.
* If the user asks too many irrelevant questions, politely redirect or conclude with a summary.
* Do not answer generic informational queries on medications, treatments, procedures, or conditions.
* **Before Summarizing:** Ask: "Do you have any other {procedure} symptoms or concerns to share with the doctor?" Gather details if 'yes', then summarize.
* **Summary Format:** "Thank you. Based on what you've shared, I will summarize the details gathered: ..." (Fill relevant fields only).

**Retrieved Context (General Guidance Only):**
{context}

**Conversation History (Use this to decide the NEXT question):**
{history}

**Current User Input:** {input}

**Your Response (Next Question or Summary):**"""


symptom_prompt = ChatPromptTemplate.from_messages([
    ("system", SYMPTOM_SYSTEM_PROMPT),
    # MessagesPlaceholder(variable_name="messages") # History passed explicitly
])

# Chain for symptom collection
symptom_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(x["context"]),
        history=lambda x: format_history(x["messages"][:-1]), # Pass history excluding current input
        input=lambda x: x["messages"][-1].content, # Isolate current input
        # Provide config values
        clinic_name=lambda _: CLINIC_CONFIG["clinic_name"],
        procedure=lambda _: CLINIC_CONFIG["procedure"],
        doctor_name=lambda _: CLINIC_CONFIG["doctor_name"]
    )
    | symptom_prompt
    | llm
    | StrOutputParser()
)

# 3. Follow-Up Bot (No changes needed here for repetition issue)
FOLLOWUP_SYSTEM_PROMPT = """You are a post-appointment follow-up assistant for {clinic_name}.
Use the patient's prescription details:
{prescription}

Conversation History:
{history}

Ask about:
1. Medication adherence
2. Side effects
3. Symptom changes
4. Understanding of instructions

Rules:
- Ask one question at a time.
- Start the follow-up (only the *first* time you enter this state for the conversation) with: "Alright, let's discuss your follow-up. Based on your prescription, have you been taking your medications as prescribed?"
- If continuing the follow-up conversation, ask the next relevant question based on the history. **DO NOT REPEAT QUESTIONS.**
- Never modify the prescription or give medical advice.
- Escalate complex issues or new symptoms by suggesting the user contact the clinic directly.

Current Input: {input}
Next Question:"""


followup_prompt = ChatPromptTemplate.from_messages([
    ("system", FOLLOWUP_SYSTEM_PROMPT),
    # MessagesPlaceholder(variable_name="messages") # History passed explicitly
])

followup_chain = (
    RunnablePassthrough.assign(
        prescription=lambda x: x.get("prescription", "No prescription details available."), # Use state's prescription
        history=lambda x: format_history(x["messages"][:-1]), # Pass history excluding current input
        input=lambda x: x["messages"][-1].content, # Isolate current input
        clinic_name=lambda _: CLINIC_INFO["name"] # Use general clinic name
    )
    | followup_prompt
    | llm
    | StrOutputParser()
)


# ---- Cell 17 ----
# ====================
# Graph Nodes
# ====================

def get_info_node(state: ChatState):
    """Node to handle general information requests."""
    print("--- Executing Get Info Node ---")
    query = state["messages"][-1].content
    # Retrieve context using the doctor_retriever
    context_docs = doctor_retriever.invoke(query) if doctor_retriever else []
    # Invoke the chain with the current state messages and retrieved context
    response = get_info_chain.invoke({
        "messages": state["messages"],
        "context": context_docs
    })
    return {"messages": [AIMessage(content=response)], "context": context_docs} # Update context in state if needed

def symptom_node(state: ChatState):
    """Node to handle symptom collection."""
    print("--- Executing Symptom Node ---")
    query = state["messages"][-1].content
    # Retrieve context from symptom retrievers
    context_dim = retriever_dim.invoke(query) if retriever_dim else []
    context_cls = retriever_cls.invoke(query) if retriever_cls else []
    combined_context = context_dim + context_cls
    # Invoke the symptom chain with the full message history and combined context
    response = symptom_chain.invoke({
        "messages": state["messages"], # Pass the full message history
        "context": combined_context
        # Clinic config is now passed within the chain setup
    })
    return {"messages": [AIMessage(content=response)], "context": combined_context} # Update context

def followup_node(state: ChatState):
    """Node to handle post-appointment follow-up."""
    print("--- Executing Follow-up Node ---")
    # Invoke the follow-up chain with the full message history and prescription from state
    response = followup_chain.invoke({
        "messages": state["messages"], # Pass the full message history
        "prescription": state.get("prescription", SAMPLE_PRESCRIPTION) # Get prescription from state
    })
    return {"messages": [AIMessage(content=response)], "context": None} # No specific context retrieved here


def clarify_node(state: ChatState):
    """Node to ask the clarification question."""
    print("--- Executing Clarify Node ---")
    prompt = f"""**Welcome to {CLINIC_INFO['name']}!**
To help direct your query, please tell me if you are contacting us about:
1. General clinic information (hours, services, directions)
2. Current symptoms / New appointment query
3. Follow-up from a recent visit

Please respond with the number (1, 2, or 3)."""
    # Set the flag indicating clarification is needed
    return {"messages": [AIMessage(content=prompt)], "needs_clarification": True, "context": None}

def process_clarification_node(state: ChatState):
    """Node to process the user's response to the clarification question."""
    print("--- Executing Process Clarification Node ---")
    last_message = state["messages"][-1].content.strip()
    status = None
    follow_up_message = ""

    if last_message == "1":
        status = "pre"
        follow_up_message = "Great, what information about the clinic are you looking for?"
    elif last_message == "2":
        status = "during"
        # The symptom node will handle the specific welcome for symptom collection
        # follow_up_message = f"Okay, let's discuss your symptoms for {CLINIC_CONFIG['doctor_name']}. Can you start by telling me what {CLINIC_CONFIG['procedure']} symptoms bring you in today?"
    elif last_message == "3":
        status = "post"
        # The follow-up node will handle the specific welcome for follow-up
        # follow_up_message = "Alright, let's discuss your follow-up. Based on your prescription, have you been taking your medications as prescribed?"
    else:
        # Invalid response, ask again by setting needs_clarification
         return {
             "messages": [AIMessage(content="I'm sorry, I didn't quite understand that. Please reply with just the number 1, 2, or 3.")],
             "needs_clarification": True, # Signal to ask again
             "patient_status": None # Keep status as None
         }

    # Valid response, update status and clear flag
    # No need to return a follow_up_message here, the graph will route to the correct node which generates its own message.
    return {
        "patient_status": status,
        "needs_clarification": False,
        # "messages": [AIMessage(content=follow_up_message)] if follow_up_message else [] # Let the next node handle the message
    }


# ---- Cell 18 ----
# ====================
# Graph Edges & Routing Logic
# ====================


def route_logic(state: ChatState, appointment_data: dict) -> Literal["clarify", "process_clarify", "get_info", "symptom", "followup"]:
    """ Determines the next node to execute based on appointment data."""
    print(f"--- Routing Logic: Status='{state.get('patient_status')}', Needs Clarification='{state.get('needs_clarification')}' ---")

    # Check if the state indicates clarification was just asked (needs_clarification=True)
    if state.get("needs_clarification"):
        return "process_clarify"
    
    # Get current time
    current_time = datetime.now(timezone.utc)
    
    # Extract relevant appointments
    future_appointments = [
        appt for appt in appointment_data.get("appointments", [])
        if appt["appt-status"] == "booked" and 
           (datetime.fromisoformat(appt["appt-datetime"]) - current_time).total_seconds() < 48*3600
    ]
    past_appointments = [
        appt for appt in appointment_data.get("appointments", [])
        if appt["appt-status"] == "completed"
    ]

    # Routing logic based on appointment data
    if not past_appointments and not future_appointments:
        # No appointments found - use Q&A bot
        print("Routing -> get_info")
        return "get_info"
    
    if future_appointments:
        if not past_appointments:
            # First time visitor with future appointment - start symptom collection
            print("Routing -> symptom")
            return "symptom"
        else:
            # Check if same episode
            same_episode = classifier_chain.invoke({
                "query": f"Is this appointment part of the same episode as previous? Current symptoms: {appointment_data.get('current_symptoms')}, Previous summary: {past_appointments[0].get('symptom-summary')}"
            }).strip().lower()
            
            if same_episode == "yes":
                # Use previous summary and prescription
                state["prescription"] = past_appointments[0].get("prescrption")
                state["symptom-summary"] = past_appointments[0].get("symptom-summary")
            print("Routing -> symptom")
            return "symptom"
    
    if past_appointments and not future_appointments:
        # Past appointment with no future appointments - use followup bot
        state["prescription"] = past_appointments[0].get("prescrption")
        print("Routing -> followup")
        return "followup"

    # Default to clarification
    print("Routing -> clarify (No matching condition)")
    return "clarify"

# ---- Cell 19 ----
# ====================
# State Graph Definition
# ====================
workflow = StateGraph(ChatState)

# Define nodes
workflow.add_node("clarify", clarify_node)
workflow.add_node("process_clarify", process_clarification_node)
workflow.add_node("get_info", get_info_node)
workflow.add_node("symptom", symptom_node)
workflow.add_node("followup", followup_node)

# Define edges
workflow.set_conditional_entry_point(
    route_logic,
    {
        "clarify": "clarify",
        "process_clarify": "process_clarify", # Should not happen on entry, but for completeness
        "get_info": "get_info",
        "symptom": "symptom",
        "followup": "followup",
    }
)

# After asking for clarification, wait for user input
workflow.add_edge("clarify", END)

# After processing clarification, route to the correct bot node
workflow.add_conditional_edges(
    "process_clarify",
    # Route based on the status determined in process_clarification_node
    lambda state: state.get("patient_status") if state.get("patient_status") else "clarify",
    {
        "pre": "get_info",
        "during": "symptom",
        "post": "followup",
        "clarify": "clarify" # If processing failed, ask again
    }
)

# After a bot node finishes, END the current run (wait for next user input)
workflow.add_edge("get_info", END)
workflow.add_edge("symptom", END)
workflow.add_edge("followup", END)


# ====================
# Compile the Graph with Memory
# ====================
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
print("Graph compiled successfully.")



# ---- Cell 21 ----
# ====================
# Conversation Execution Logic (Corrected)
# ====================
def run_conversation():
    """ Runs the chatbot conversation loop."""
    thread_id = input("Enter a unique thread ID for this conversation (e.g., patient123): ")
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\nStarting/Resuming conversation with thread ID: {thread_id}")
    print("Type 'exit' or 'quit' to end the conversation.")

    # Get appointment data
    appointment_data = get_appointment_data(thread_id)
    
    # Get the initial state from the checkpointer or start fresh
    current_state = app.get_state(config)

    if current_state and current_state.values.get("messages"):
        last_ai_message = next((msg for msg in reversed(current_state.values["messages"]) if isinstance(msg, AIMessage)), None)
        if last_ai_message:
            print(f"\nAssistant: {last_ai_message.content}")
        else:
            initial_state = app.invoke({}, config)
            if initial_state and initial_state.get("messages"):
                print(f"\nAssistant: {initial_state['messages'][-1].content}")
            else:
                print("\nAssistant: Hello! How can I help you today?")
    else:
        initial_state = app.invoke({}, config)
        if initial_state and initial_state.get("messages"):
            print(f"\nAssistant: {initial_state['messages'][-1].content}")
        else:
            print("\nAssistant: Hello! How can I help you today?")

    while True:
        query = input("Patient: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("\nAssistant: Goodbye!")
            break

        input_data = {"messages": [HumanMessage(content=query)]}
        state = app.invoke(input_data, config)
        
        if state and state.get("messages"):
            last_ai_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
            if last_ai_message:
                print(f"\nAssistant: {last_ai_message.content}")
            else:
                print("\nAssistant: (No response generated)")
        else:
            print("\nAssistant: I encountered an issue processing that. Could you please try again?")



# ---- Cell 22 ----
if __name__ == "__main__":
    # Ensure RAG data setup is complete before starting conversation
    # Check if retrievers were initialized successfully
    if retriever_dim and retriever_cls and doctor_retriever:
        print("\nAll retrievers initialized. Starting conversation loop...")
        run_conversation()
    else:
        print("\nCould not initialize all data retrievers. Exiting.")
        if not retriever_dim: print("- Symptom dimension retriever failed.")
        if not retriever_cls: print("- Symptom classification retriever failed.")
        if not doctor_retriever: print("- Doctor info retriever failed.")


