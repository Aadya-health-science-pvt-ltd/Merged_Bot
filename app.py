# app.py

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig  # Import RunnableConfig

import lance_main  # Import lance_main

load_dotenv()

app = Flask(__name__)

executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

conversations = {}
SESSION_TIMEOUT = timedelta(minutes=15)


@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    """
    Start a new conversation thread with optional context parameters.
    """
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    doctor_name = data.get('doctor_name')
    if not thread_id or not doctor_name:
        return jsonify({'error': 'thread_id and doctor_name are required'}), 400

    cfg = {
        'thread_id': thread_id,
        'doctor_name': doctor_name,
        'clinic_name': data.get('clinic_name'),
        'specialty': data.get('specialty', 'paediatrics'),
        'appointment_data': data.get('appointment_data'),
        'history': []
    }

    # Initialize retrievers
    doctor_retriever = lance_main.setup_doctor_info_retriever(cfg['specialty'])
    retriever_dim = lance_main.vector_store_dim
    retriever_cls = lance_main.vector_store_cls

    if doctor_retriever and retriever_dim and retriever_cls:
        conversations[thread_id] = {
            'app': lance_main.app,  # Store the LangGraph app
            'last_activity': datetime.now(timezone.utc),
            'configurable': cfg,
            'appointment_data': data.get('appointment_data', {})
        }
        return jsonify({'message': f'Conversation {thread_id} started with doctor: {cfg["doctor_name"]} and specialty: {cfg["specialty"]}.'}), 200
    else:
        return jsonify({'error': 'Failed to initialize retrievers.'}), 500


@app.route('/message', methods=['POST'])
def send_message():
    """
    Send a user message to the chatbot.
    """
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    user_message = data.get('message')
    appointment_data = data.get('appointment_data')

    if not thread_id or not user_message:
        return jsonify({'error': 'thread_id and message are required'}), 400

    if thread_id not in conversations:
        return jsonify({'error': 'Conversation not found. Call /start_conversation first.'}), 404

    conv = conversations[thread_id]
    if datetime.now(timezone.utc) - conv['last_activity'] > SESSION_TIMEOUT:
        conversations.pop(thread_id, None)
        return jsonify({'error': 'Session expired after 15 minutes of inactivity.'}), 440

    conv['last_activity'] = datetime.now(timezone.utc)

    configurable = conv['configurable'].copy()
    if data.get('prescription'):
        configurable['prescription'] = data['prescription']

    # Update appointment_data
    if appointment_data:
        conversations[thread_id]['appointment_data'] = appointment_data
        configurable['appointment_data'] = appointment_data

    route_config = {
        'doctor_name': configurable.get('doctor_name'),
        'appointment_data': conversations[thread_id]['appointment_data'],
        'patient_status': conv['configurable'].get('patient_status'),
        'history': conv['configurable'].get('history', []) + [HumanMessage(content=user_message)]
    }
    config = {'configurable': configurable, 'route_config': route_config}

    print(f"Config being passed to route_logic: {config}")
    current_messages = conv['configurable'].get('history', []) + [HumanMessage(content=user_message)]
    # Construct the ChatState.
    input_data = lance_main.ChatState(
        messages=current_messages,
        patient_status=conv['configurable'].get('patient_status'),
        appointment_data=conversations[thread_id]['appointment_data']
    )
    print(f"input_data: {input_data}")

    bot_selection_message = None
    if not conv['configurable'].get('has_interacted', False):
        route = lance_main.route_logic(
            input_data,  # Pass the ChatState
            RunnableConfig(config=config)
        )
        conv['configurable']['has_interacted'] = True
        if route == "get_info":
            bot_selection_message = "Get Info Bot selected."
        elif route == "symptom":
            bot_selection_message = "Symptom Bot selected."
        elif route == "followup":
            bot_selection_message = "Follow-up Bot selected."
        elif route == "same_episode_check":
            bot_selection_message = "Same Episode Check initiated."
        print(f"Route selected by route_logic: {route}")

    future = executor.submit(lambda: conv['app'].invoke(input_data, config))
    state = future.result()
    print(f"state after invoke: {state}")

    reply = ''
    if state and state.get('messages'):
        for msg in reversed(state['messages']):
            if isinstance(msg, AIMessage):
                reply = msg.content
                break

    # Update history.  Important to use the history from the state.
    conv['configurable']['history'] = state.get('messages', current_messages) #  Get messages from state
    if reply:
        conv['configurable']['history'].append(AIMessage(content=reply))

    if state and state.get('same_episode_response'):
        conversations[thread_id]['same_episode_response'] = state.get('same_episode_response')

    response_data = {'reply': reply}
    if bot_selection_message:
        response_data['bot_selection'] = bot_selection_message

    return jsonify(response_data), 200


if __name__ == '__main__':
    port = int(os.getenv('FLASK_RUN_PORT', 5007))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=True)
