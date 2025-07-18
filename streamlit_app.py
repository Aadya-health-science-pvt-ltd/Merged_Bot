import streamlit as st
import requests
import uuid
from datetime import datetime
import os

# Backend URL (adjust if needed)
BACKEND_URL = os.environ.get("BACKEND_URL", "https://merged-bot.onrender.com")

st.set_page_config(page_title="Medical Assistant Bot", page_icon="💬", layout="centered")
st.title("💬 Medical Assistant Bot (Streamlit)")

# Session state for Q&A wizard
if 'qa_step' not in st.session_state:
    st.session_state.qa_step = 0
if 'qa_answers' not in st.session_state:
    st.session_state.qa_answers = {}
if 'appointment_list' not in st.session_state:
    st.session_state.appointment_list = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}

# Q&A steps
def qa_wizard():
    steps = [
        ("phone_number", "Enter your phone number (will be used as your session ID):", ""),
        ("doctor_name", "Enter the doctor's name:", "Dr. Balachandra BV"),
        ("age", "Enter the patient's age (in months or years):", ""),
        ("gender", "Select the patient's gender:", "male"),
        ("consultation_type", "Enter the consultation type:", "Child Allergy and Asthma Consultation"),
        ("specialty", "Enter the medical specialty:", "paediatrics"),
        ("clinic_name", "Enter the clinic name:", "Chirayu clinic"),
        ("symptoms", "What is the main symptom or reason for this visit?", ""),
        ("add_appointment", "Add an appointment?", None),
    ]
    step = st.session_state.qa_step
    answers = st.session_state.qa_answers
    appointments = st.session_state.appointment_list

    if step < len(steps):
        key, question, default = steps[step]
        st.subheader(f"Step {step+1} of {len(steps)}")
        if key == "add_appointment":
            add = st.radio(question, ["Yes", "No"], key=key)
            if add == "Yes":
                appt_doctor = st.text_input("Appointment doctor name", answers.get("doctor_name", ""), key="appt_doctor")
                appt_datetime = st.text_input("Appointment datetime (YYYY-MM-DDTHH:MM:SSZ)", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), key="appt_datetime")
                appt_status = st.selectbox("Appointment status", ["booked", "completed"], key="appt_status")
                if st.button("Add Appointment"):
                    appointments.append({
                        "doctor_name": appt_doctor,
                        "appt_datetime": appt_datetime,
                        "appt_status": appt_status
                    })
                    st.success("Appointment added!")
                st.write("Current Appointments:")
                st.json(appointments)
            if st.button("Next"):
                st.session_state.qa_step += 1
                st.rerun()
            return
        elif key == "gender":
            val = st.selectbox(question, ["male", "female", "other"], key=key)
        else:
            val = st.text_input(question, default, key=key)
        if st.button("Next"):
            if key == "phone_number" and not val.strip():
                st.warning("Phone number is required to proceed.")
                return
            answers[key] = val
            st.session_state.qa_step += 1
            st.rerun()
    else:
        # Show summary and confirm
        st.subheader("Summary of your details:")
        thread_id = answers["phone_number"].strip()
        payload = {
            "thread_id": thread_id,
            "phone_number": thread_id,
            "doctor_name": answers["doctor_name"],
            "age": answers["age"],
            "gender": answers["gender"],
            "consultation_type": answers["consultation_type"],
            "specialty": answers["specialty"],
            "clinic_name": answers["clinic_name"],
            "symptoms": answers["symptoms"] if answers["symptoms"] and answers["symptoms"].strip() and answers["symptoms"].lower() != "none" else "",
            "appointment_data": {"appointments": appointments}
        }
        st.json(payload)
        if st.button("Start Conversation"):
            try:
                resp = requests.post(f"{BACKEND_URL}/start_conversation", json=payload)
                if resp.status_code == 200:
                    st.session_state.thread_id = thread_id
                    st.session_state.conversation_started = True
                    st.session_state.patient_info = payload
                    st.session_state.messages = []  # Start with empty message list
                    st.success("Conversation started!")
                    # Check if there are appointments to determine if symptoms are required
                    has_appointments = payload.get("appointment_data", {}).get("appointments", [])
                    
                    # If no appointments, allow starting without symptoms (will route to get_info)
                    if not has_appointments:
                        # Start with a general greeting to trigger get_info bot
                        initial_message = "Hello"
                        initial_payload = {
                            "thread_id": thread_id,
                            "age": payload.get("age"),
                            "gender": payload.get("gender"),
                            "vaccine_visit": "yes" if "vaccine" in payload.get("consultation_type", "").lower() else "no",
                            "message": initial_message,
                            "specialty": payload.get("specialty"),
                            "message_type": "human"
                        }
                    else:
                        # If there are appointments, require symptoms
                    initial_message = payload.get("symptoms", "")
                    if not initial_message:
                        st.warning("Symptom is required to start the conversation.")
                        return
                    initial_payload = {
                        "thread_id": thread_id,
                        "age": payload.get("age"),
                        "gender": payload.get("gender"),
                        "vaccine_visit": "yes" if "vaccine" in payload.get("consultation_type", "").lower() else "no",
                        "symptoms": initial_message,
                        "message": initial_message,
                        "specialty": payload.get("specialty"),
                        "message_type": "human"
                    }
                    try:
                        resp_msg = requests.post(f"{BACKEND_URL}/message", json=initial_payload)
                        if resp_msg.status_code == 200:
                            reply = resp_msg.json().get("reply", "")
                            st.session_state.messages.append({"type": "bot", "content": reply})
                        else:
                            st.session_state.messages.append({"type": "bot", "content": f"Error: {resp_msg.json().get('error', resp_msg.text)}"})
                    except Exception as e:
                        st.session_state.messages.append({"type": "bot", "content": f"Error: {e}"})
                    # Immediately send two dummy messages to get initial bot selection (optional, can be removed if not needed)
                    dummy_payload_1 = {
                        "thread_id": thread_id,
                        "message": "dummy",
                        "specialty": payload.get("specialty"),
                        "message_type": "human"
                    }
                    dummy_payload_2 = {
                        "thread_id": thread_id,
                        "message": "dummy",
                        "specialty": payload.get("specialty"),
                        "message_type": "human"
                    }
                    try:
                        dummy_resp_1 = requests.post(f"{BACKEND_URL}/message", json=dummy_payload_1)
                        dummy_resp_2 = requests.post(f"{BACKEND_URL}/message", json=dummy_payload_2)
                        if dummy_resp_2.status_code == 200:
                            st.session_state.last_bot_selection = dummy_resp_2.json().get("bot_selection")
                        else:
                            st.session_state.last_bot_selection = None
                    except Exception:
                        st.session_state.last_bot_selection = None
                else:
                    st.error(f"Failed to start conversation: {resp.json().get('error', resp.text)}")
            except Exception as e:
                st.error(f"Error: {e}")
        if st.button("Restart Q&A"):
            for key in ["qa_step", "qa_answers", "appointment_list"]:
                st.session_state.pop(key, None)
            st.rerun()

# Main logic
if not st.session_state.conversation_started:
    qa_wizard()

# Chat interface
if st.session_state.conversation_started:
    st.subheader("Chat with Medical Assistant")
    if 'last_bot_selection' not in st.session_state:
        st.session_state.last_bot_selection = None
    if 'doctor_info_url' not in st.session_state:
        st.session_state.doctor_info_url = ""
    if 'website_embedded' not in st.session_state:
        st.session_state.website_embedded = False
    if 'embedding_in_progress' not in st.session_state:
        st.session_state.embedding_in_progress = False
    show_doctor_info_url = (
        st.session_state.last_bot_selection is not None and
        (
            'get info' in st.session_state.last_bot_selection.lower() or
            'get_info' in st.session_state.last_bot_selection.lower()
        )
    )
    if show_doctor_info_url:
        st.markdown("**If you want to ask about the doctor or clinic, paste the website link below (required before Q&A):**")
        st.session_state.doctor_info_url = st.text_input(
            "Doctor's website link (for info queries)",
            st.session_state.doctor_info_url,
            key="doctor_info_url_chat"
        )
        if st.button("Load Website and Embed", disabled=st.session_state.embedding_in_progress):
            if st.session_state.doctor_info_url.strip():
                st.session_state.embedding_in_progress = True
                with st.spinner("Embedding website content. Please wait..."):
                    import requests
                    try:
                        resp = requests.post(f"{BACKEND_URL}/embed_website", json={"url": st.session_state.doctor_info_url.strip()})
                        if resp.status_code == 200 and resp.json().get("success"):
                            st.session_state.website_embedded = True
                            st.success("Website embedded successfully! You can now ask info questions.")
                        else:
                            st.session_state.website_embedded = False
                            st.error(f"Embedding failed: {resp.json().get('error', resp.text)}")
                    except Exception as e:
                        st.session_state.website_embedded = False
                        st.error(f"Embedding failed: {e}")
                st.session_state.embedding_in_progress = False
        if not st.session_state.website_embedded:
            st.info("You must load and embed the website before asking info questions.")
    prescription_to_send = None
    prescription_str = ""
    show_prescription = (
        st.session_state.last_bot_selection is not None and
        'followup' in st.session_state.last_bot_selection.lower()
    )
    if show_prescription:
        st.markdown("**If this is a follow-up, paste the prescription JSON below (optional):**")
        prescription_str = st.text_area("Prescription JSON (for follow-up only)", "", key="prescription_json_chat")
        if prescription_str.strip():
            try:
                import json
                prescription_obj = json.loads(prescription_str)
                prescription_to_send = json.dumps(prescription_obj, indent=2)
            except Exception:
                prescription_to_send = prescription_str.strip()
    for msg in st.session_state.messages:
        if msg["type"] == "user":
            st.markdown(f"<div style='text-align: right; color: #2563eb;'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: #111827;'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
    user_input = st.text_area("Your message", key="user_input", height=70, disabled=(show_doctor_info_url and not st.session_state.website_embedded))
    if st.button("Send") and user_input.strip() and (not show_doctor_info_url or st.session_state.website_embedded):
        st.session_state.messages.append({"type": "user", "content": user_input})
        patient_info = st.session_state.patient_info
        payload = {
            "thread_id": st.session_state.thread_id,
            "age": patient_info.get("age"),
            "gender": patient_info.get("gender"),
            "vaccine_visit": "yes" if "vaccine" in patient_info.get("consultation_type", "").lower() else "no",
            "message": user_input,
            "specialty": patient_info.get("specialty"),
            "message_type": "human"
        }
        
        # Only set symptoms if the user is actually providing symptoms
        # This should be determined by the current bot context or user intent
        # For now, we'll only set symptoms if the last bot selection was symptom-related
        # OR if this is the first message and user is clearly describing symptoms
        should_set_symptoms = False
        
        # Check if we're in a symptom context
        if (st.session_state.last_bot_selection and 
            ('symptom' in st.session_state.last_bot_selection.lower() or
             'followup' in st.session_state.last_bot_selection.lower())):
            should_set_symptoms = True
        
        # Check if this is the first message and user is clearly describing symptoms
        elif (not st.session_state.last_bot_selection and 
              any(symptom_word in user_input.lower() for symptom_word in 
                   ['fever', 'cough', 'cold', 'pain', 'vomit', 'diarrhea', 'rash', 'headache', 'stomach', 'ear', 'throat'])):
            should_set_symptoms = True
        
        if should_set_symptoms:
            payload["symptoms"] = user_input
        # Always include doctor_info_url for info queries after embedding
        if show_doctor_info_url and st.session_state.doctor_info_url.strip():
            payload["doctor_info_url"] = st.session_state.doctor_info_url.strip()
        if show_prescription and prescription_to_send is not None:
            payload["prescription"] = prescription_to_send
        print("Sending payload to /message:", payload)  # Debug print
        try:
            resp = requests.post(f"{BACKEND_URL}/message", json=payload)
            if resp.status_code == 200:
                reply = resp.json().get("reply", "(No reply)")
                st.session_state.messages.append({"type": "bot", "content": reply})
                # Store bot_selection for next turn
                st.session_state.last_bot_selection = resp.json().get("bot_selection")
            else:
                st.session_state.messages.append({"type": "bot", "content": f"Error: {resp.json().get('error', resp.text)}"})
        except Exception as e:
            st.session_state.messages.append({"type": "bot", "content": f"Error: {e}"})
        st.rerun()
    if st.button("Restart Conversation"):
        for key in ["thread_id", "conversation_started", "messages", "patient_info", "qa_step", "qa_answers", "appointment_list", "last_bot_selection", "doctor_info_url", "website_embedded", "embedding_in_progress"]:
            st.session_state.pop(key, None)
        st.rerun() 