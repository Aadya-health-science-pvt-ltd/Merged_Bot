import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from utils.general_utils import extract_specialty_and_age, build_or_load_faiss
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Optional, Literal # Make sure Literal is here for decide_bot_route return type
import typing

import lance_main  # Import everything from lance_main
from conversation.chat_state import initialize_symptom_session

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 2) * 2)

conversations = {}
SESSION_TIMEOUT = timedelta(minutes=15) # Session expires after 15 minutes of inactivity


@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    """
    Start a new conversation thread with optional context parameters.
    Initializes a conversation session, storing references to the compiled bot apps
    and setting up initial conversation state.
    """
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    doctor_name = data.get('doctor_name')
    consultation_type = data.get('consultation_type')
    specialty = data.get('specialty')
    age_group = data.get('age_group')
    age = data.get('age')
    vaccine_visit = data.get('vaccine_visit')
    symptoms = data.get('symptoms')

    # If consultation_type is provided, extract specialty and age_group from it
    if consultation_type:
        specialty, age_group = extract_specialty_and_age(consultation_type)
    else:
        # fallback to provided values or defaults
        specialty = specialty or 'paediatrics'
        age_group = age_group or None
    
    if not thread_id or not doctor_name:
        return jsonify({'error': 'thread_id and doctor_name are required'}), 400

    # Initialize the conversation state for the new thread_id
    conversations[thread_id] = {
        'get_info_app': lance_main.get_info_app,   # Reference to the compiled Get Info bot
        'symptom_app': lance_main.symptom_app,     # Reference to the compiled Symptom bot
        'followup_app': lance_main.followup_app,   # Reference to the compiled Follow-up bot
        'last_activity': datetime.now(timezone.utc), # Timestamp for session timeout
        'configurable': {
            'thread_id': thread_id,
            'doctor_name': doctor_name,
            'clinic_name': data.get('clinic_name', lance_main.CLINIC_INFO["name"]),
            'specialty': specialty,
            'age_group': age_group,
            'age': age,
            'gender': data.get('gender'),
            'consultation_type': consultation_type,
            'vaccine_visit': vaccine_visit,
            'current_thread_history': [],
            'is_initial_message': True,
            'current_bot_key': None,
            'ask_same_episode': False,
            'prescription': data.get('prescription', lance_main.SAMPLE_PRESCRIPTION),
            'symptom_summary': None,
            'doctor_info_url': data.get('doctor_info_url', None),
            'services': data.get('services', ""),
            'symptoms': symptoms,
        },
        'appointment_data': data.get('appointment_data', {}) # Store appointment details
    }
    
    return jsonify({
        'message': f'Conversation {thread_id} started with doctor: {doctor_name} and specialty: {conversations[thread_id]["configurable"]["specialty"]}.'
    }), 200


@app.route('/message', methods=['POST'])
def send_message():
    """
    Sends a user message to the chatbot. This endpoint handles routing to the correct bot
    and managing the conversation history.
    """
    data = request.get_json() or {}
    print("Received /message payload:", data)  # Debug print
    thread_id = data.get('thread_id')
    user_message = data.get('message')

    if not thread_id or not user_message:
        return jsonify({'error': 'thread_id and message are required'}), 400

    if thread_id not in conversations:
        return jsonify({'error': 'Conversation not found. Call /start_conversation first.'}), 404

    conv = conversations[thread_id]

    # Check for session timeout
    if datetime.now(timezone.utc) - conv['last_activity'] > SESSION_TIMEOUT:
        conversations.pop(thread_id, None) # Remove expired session
        return jsonify({'error': 'Session expired after 15 minutes of inactivity.'}), 440

    conv['last_activity'] = datetime.now(timezone.utc) # Update last activity timestamp

    # Update all dynamic fields from payload BEFORE constructing ChatState
    if data.get('appointment_data'):
        conv['appointment_data'] = data['appointment_data']
    if data.get('prescription'):
        conv['configurable']['prescription'] = data['prescription']
    if data.get('doctor_info_url'):
        conv['configurable']['doctor_info_url'] = data['doctor_info_url']
    if data.get('clinic_name'):
        conv['configurable']['clinic_name'] = data['clinic_name']
    if data.get('doctor_name'):
        conv['configurable']['doctor_name'] = data['doctor_name']
    if data.get('services'):
        conv['configurable']['services'] = data['services']
    for key in ['age_group', 'age', 'gender', 'specialty', 'vaccine_visit', 'symptoms']:
        if data.get(key) is not None:
            conv['configurable'][key] = data[key]

    # Add symptoms to state if present
    if data.get('symptoms'):
        conv['configurable']['symptoms'] = data['symptoms']

    user_message_obj = HumanMessage(content=user_message)
    
    # Append the new user message to the thread's cumulative history
    conv['configurable']['current_thread_history'].append(user_message_obj)

    selected_app = None # This will hold the LangGraph app to invoke (e.g., lance_main.symptom_app)
    bot_selection_message = None # Message indicating which bot was selected
    reply = '' # The AI's response

    # RunnableConfig for LangGraph's checkpointer (memory) and other configurable parameters
    runnable_config_obj = RunnableConfig(configurable={
        "thread_id": thread_id, # CRITICAL: This links to the MemorySaver for each bot
        "doctor_name": conv['configurable']['doctor_name'],
        "clinic_name": conv['configurable']['clinic_name'],
        "specialty": conv['configurable']['specialty']
    })

    # Ensure appointment_data is present for routing
    if not conv.get('appointment_data') or not conv['appointment_data']:
        conv['appointment_data'] = conversations[thread_id].get('appointment_data', {})
    print('[DEBUG] appointment_data before routing:', conv['appointment_data'])

    # Ensure appointment_data is present in the configurable state for routing
    conv['configurable']['appointment_data'] = conv['appointment_data']

    # --- Routing Logic: Prioritized If-Else Structure ---
    print(f"[DEBUG] Routing logic - ask_same_episode: {conv['configurable']['ask_same_episode']}")
    print(f"[DEBUG] Routing logic - is_initial_message: {conv['configurable']['is_initial_message']}")
    print(f"[DEBUG] Routing logic - current_bot_key: {conv['configurable']['current_bot_key']}")

    # Priority 1: Handling "Same Episode" follow-up question (Rule 3's second step)
    if conv['configurable']['ask_same_episode']:
        user_response = user_message.lower().strip()
        conv['configurable']['ask_same_episode'] = False # Reset the flag after processing the response

        if user_response == 'yes':
            # User confirmed it's the same episode, retrieve previous details
            previous_appointment = next((appt for appt in conv['appointment_data'].get("appointments", []) if appt.get("appt_status") == "completed" and appt.get("doctor_name") == conv['configurable']['doctor_name']), None)
            if previous_appointment:
                # Update current state with previous prescription and symptom summary
                prescription = previous_appointment.get("prescription", None)
                symptom_summary = previous_appointment.get("symptom-summary", None)
                conv['configurable']['prescription'] = prescription
                conv['configurable']['symptom_summary'] = symptom_summary
                print("User confirmed same episode. Routing to Symptom Bot with previous context.")
            else:
                print("Previous appointment details not found for continuity. Starting fresh symptom collection.")
            conv['configurable']['current_bot_key'] = 'symptom' # Set symptom bot as active
            return jsonify({'reply': 'Continuing with previous episode. Please describe any new symptoms or concerns.'}), 200
        else: # user_response is 'no' or anything else
            print("User indicated new episode. Routing to Symptom Bot for fresh collection.")
            conv['configurable']['current_bot_key'] = 'symptom' # Set symptom bot as active
            return jsonify({'reply': 'Starting a new episode. Please describe your current symptoms.'}), 200

    # Priority 2: Initial routing (first message in a new conversation thread)
    elif conv['configurable']['is_initial_message']:
        print("[DEBUG] Initial routing - calling decide_bot_route")
        conv['configurable']['is_initial_message'] = False # Mark as not initial anymore

        # Ensure 'messages' key is present for routing
        conv['configurable']['messages'] = conv['configurable']['current_thread_history']
        # Use the external router function from lance_main to decide the initial bot
        route_decision = lance_main.decide_bot_route(conv['configurable'], runnable_config_obj)
        print(f"[DEBUG] Router decision: {route_decision}")
        
        if route_decision == "get_info":
            selected_app = conv['get_info_app']
            bot_selection_message = "Get Info Bot selected."
        elif route_decision == "symptom":
            selected_app = conv['symptom_app']
            bot_selection_message = "Symptom Bot selected."
        elif route_decision == "followup":
            selected_app = conv['followup_app']
            bot_selection_message = "Follow-up Bot selected."
        elif route_decision == "same_episode_check":
            conv['configurable']['ask_same_episode'] = True # Set flag to ask "same episode?" next
            # Immediately return the question to the user, no bot invoked yet
            return jsonify({
                'reply': f"Is this related to your previous visit with Dr. {conv['configurable']['doctor_name']}? Please answer 'yes' or 'no'.",
                'bot_selection': "Same Episode Check initiated."
            }), 200
        else: # Fallback for unexpected route_decision (should ideally not happen)
            selected_app = conv['get_info_app']
            bot_selection_message = "Defaulting to Get Info Bot."
        
        conv['configurable']['current_bot_key'] = route_decision # Store the key of the selected bot

    # Priority 3: Continue with the last selected bot for ongoing conversation
    else:
        # Retrieve the last chosen bot and use it for continuity
        bot_key = conv['configurable']['current_bot_key']
        if bot_key == "get_info":
            selected_app = conv['get_info_app']
        elif bot_key == "symptom":
            selected_app = conv['symptom_app']
        elif bot_key == "followup":
            selected_app = conv['followup_app']
        else:
            # Fallback if current_bot_key is somehow invalid or missing
            print(f"Warning: current_bot_key '{bot_key}' not found. Defaulting to Get Info Bot.")
            selected_app = conv['get_info_app']
            bot_selection_message = "Continuing with default bot (get_info)."

        bot_selection_message = f"Continuing with previously selected bot: {bot_key}."

    # Ensure symptom session is initialized if symptom bot is selected
    if conv['configurable'].get('current_bot_key') == 'symptom' and not conv['configurable'].get('symptom_prompt'):
        conv['configurable'] = initialize_symptom_session(conv['configurable'])
        conv['configurable']['symptom_prompt'] = conv['configurable'].get('symptom_prompt')
        print("[DEBUG] After initialize_symptom_session, symptom_prompt:", conv['configurable'].get('symptom_prompt'))
        conv['configurable']['messages'] = conv['configurable']['current_thread_history']
        conv['configurable']['messages'].append(HumanMessage(content='start'))
        conv['configurable']['messages'].append(HumanMessage(content='Which bot are you or what can you assist me with?'))

    # Ensure followup session is initialized if followup bot is selected
    from conversation.chat_state import initialize_followup_session
    if conv['configurable'].get('current_bot_key') == 'followup' and not conv['configurable'].get('followup_prompt'):
        conv['configurable'] = initialize_followup_session(conv['configurable'])
        conv['configurable']['followup_prompt'] = conv['configurable'].get('followup_prompt')
        print("[DEBUG] After initialize_followup_session, followup_prompt:", conv['configurable'].get('followup_prompt'))
        conv['configurable']['messages'] = conv['configurable']['current_thread_history']

    # Ensure 'messages' key is present for the bot state (set after any session initialization)
    conv['configurable']['messages'] = conv['configurable']['current_thread_history']

    # --- Invoke the Selected Bot's LangGraph Application ---
    if selected_app:
        print(f"Invoking {bot_selection_message} on thread: {thread_id}")
        
        # Execute the selected bot's graph in a separate thread to avoid blocking Flask
        future = executor.submit(lambda: selected_app.invoke(conv['configurable'], runnable_config_obj))
        
        state_after_invoke = future.result() # Get the result from the bot

        if state_after_invoke and state_after_invoke.get('messages'):
            # Extract the last AI message from the bot's full response history
            ai_reply_message = None
            for msg in reversed(state_after_invoke['messages']):
                if isinstance(msg, AIMessage):
                    ai_reply_message = msg
                    break
            
            if ai_reply_message:
                reply = ai_reply_message.content
                # Append the bot's AI message to the thread's cumulative history
                conv['configurable']['current_thread_history'].append(ai_reply_message)
            else:
                reply = "I'm sorry, I couldn't generate a clear response from the bot."
        else:
            reply = "I'm sorry, I couldn't get a valid response from the bot's processing."
    else:
        reply = "I'm sorry, I couldn't determine how to respond to your request."

    # Prepare the JSON response
    response_data = {'reply': reply}
    if bot_selection_message:
        response_data['bot_selection'] = bot_selection_message

    return jsonify(response_data), 200

@app.route('/embed_website', methods=['POST'])
def embed_website():
    data = request.get_json() or {}
    url = data.get('url')
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    try:
        # Always force rebuild for explicit embedding
        build_or_load_faiss(url, force_rebuild=True)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def process_quiz_wizard_submission(state):
    # ... existing logic to populate state with quiz wizard data ...
    state = initialize_symptom_session(state)
    # ... continue with routing to symptom_node or storing state ...
    return state

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print("Exception in Flask app:", traceback.format_exc())
    response = {
        "error": str(e),
        "type": type(e).__name__
    }
    return jsonify(response), 500

if __name__ == '__main__':
    # Set the Flask port from environment variable or default to 5007
    port = int(os.getenv('FLASK_RUN_PORT', 5009))
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, threaded=True, debug=True)