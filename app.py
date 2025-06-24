import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Optional, Literal # Make sure Literal is here for decide_bot_route return type

import lance_main  # Import everything from lance_main

load_dotenv()

app = Flask(__name__)

executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

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
    
    if not thread_id or not doctor_name:
        return jsonify({'error': 'thread_id and doctor_name are required'}), 400

    # Ensure LanceDB tables and retrievers are set up (idempotent operations)
    lance_main.setup_doctor_info_retriever()

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
            'specialty': data.get('specialty', 'paediatrics'),
            'age_group': data.get('age_group'),      # <-- Add this
            'gender': data.get('gender'),            # <-- Add this
            'symptom': data.get('symptom'),          # <-- Add this
            'current_thread_history': [],
            'is_initial_message': True,
            'current_bot_key': None,
            'ask_same_episode': False,
            'prescription': data.get('prescription', lance_main.SAMPLE_PRESCRIPTION),
            'symptom_summary': None
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

    # Update appointment_data if provided in the message payload (can be dynamic)
    if data.get('appointment_data'):
        conv['appointment_data'] = data['appointment_data']
    # Update prescription if provided (can be dynamic)
    if data.get('prescription'):
        conv['configurable']['prescription'] = data['prescription']

    user_message_obj = HumanMessage(content=user_message)
    
    # Append the new user message to the thread's cumulative history
    conv['configurable']['current_thread_history'].append(user_message_obj)

    selected_app = None # This will hold the LangGraph app to invoke (e.g., lance_main.symptom_app)
    bot_selection_message = None # Message indicating which bot was selected
    reply = '' # The AI's response

    # Update metadata if provided in the message payload
    for key in ['age_group', 'gender', 'specialty']:
        if data.get(key) is not None:
            conv['configurable'][key] = data[key]

    # --- Prepare ChatState and RunnableConfig for bot invocation ---
    # The ChatState passed to invoke() should contain the full history for the bot's context.
    current_chat_state = lance_main.ChatState(
        messages=conv['configurable']['current_thread_history'],
        patient_status=conv['configurable'].get('patient_status'),
        appointment_data=conv['appointment_data'],
        prescription=conv['configurable']['prescription'],
        symptom_summary=conv['configurable']['symptom_summary'],
        age_group=conv['configurable'].get('age_group'),      
        gender=conv['configurable'].get('gender'),            
        specialty=conv['configurable'].get('specialty')       
    )
    # RunnableConfig for LangGraph's checkpointer (memory) and other configurable parameters
    runnable_config_obj = RunnableConfig(configurable={
        "thread_id": thread_id, # CRITICAL: This links to the MemorySaver for each bot
        "doctor_name": conv['configurable']['doctor_name'],
        "clinic_name": conv['configurable']['clinic_name'],
        "specialty": conv['configurable']['specialty']
    })

    # --- Routing Logic: Prioritized If-Else Structure ---

    # Priority 1: Handling "Same Episode" follow-up question (Rule 3's second step)
    if conv['configurable']['ask_same_episode']:
        user_response = user_message.lower().strip()
        conv['configurable']['ask_same_episode'] = False # Reset the flag after processing the response

        if user_response == 'yes':
            # User confirmed it's the same episode, retrieve previous details
            previous_appointment = next((appt for appt in conv['appointment_data'].get("appointments", []) if appt.get("appt_status") == "completed" and appt.get("doctor_name") == conv['configurable']['doctor_name']), None)
            if previous_appointment:
                # Update current state with previous prescription and symptom summary
                current_chat_state['prescription'] = previous_appointment.get("prescription", current_chat_state['prescription'])
                current_chat_state['symptom_summary'] = previous_appointment.get("symptom-summary", current_chat_state['symptom_summary'])
                conv['configurable']['prescription'] = current_chat_state['prescription'] # Update conv state too
                conv['configurable']['symptom_summary'] = current_chat_state['symptom_summary'] # Update conv state too
                print("User confirmed same episode. Routing to Symptom Bot with previous context.")
            else:
                print("Previous appointment details not found for continuity. Starting fresh symptom collection.")
            selected_app = conv['symptom_app']
            bot_selection_message = "Symptom Bot selected (continuing previous episode)."
            conv['configurable']['current_bot_key'] = 'symptom' # Set symptom bot as active

        else: # user_response is 'no' or anything else
            print("User indicated new episode. Routing to Symptom Bot for fresh collection.")
            selected_app = conv['symptom_app']
            bot_selection_message = "Symptom Bot selected (new episode)."
            conv['configurable']['current_bot_key'] = 'symptom' # Set symptom bot as active

    # Priority 2: Initial routing (first message in a new conversation thread)
    elif conv['configurable']['is_initial_message']:
        conv['configurable']['is_initial_message'] = False # Mark as not initial anymore

        # Use the external router function from lance_main to decide the initial bot
        route_decision = lance_main.decide_bot_route(current_chat_state, runnable_config_obj)
        
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

    # --- Invoke the Selected Bot's LangGraph Application ---
    if selected_app:
        print(f"Invoking {bot_selection_message} on thread: {thread_id}")
        
        # Execute the selected bot's graph in a separate thread to avoid blocking Flask
        future = executor.submit(lambda: selected_app.invoke(current_chat_state, runnable_config_obj))
        
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

if __name__ == '__main__':
    # Set the Flask port from environment variable or default to 5007
    port = int(os.getenv('FLASK_RUN_PORT', 5009))
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, threaded=True, debug=True)