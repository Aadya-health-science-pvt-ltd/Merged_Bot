import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Import your LangGraph app (compiled in lance_main.py)
import lance_main  # Ensure lance_main.py defines and exports `app` graph

# Flask application\app = Flask(__name__)

# ThreadPool for parallel processing
executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

# In-memory conversation store:
# thread_id -> { "app": graph, "last_activity": datetime, "configurable": {...} }
conversations = {}
SESSION_TIMEOUT = timedelta(minutes=15)

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    """
    Start a new conversation thread with optional context parameters.
    Request JSON should include:
      - thread_id (str)
      - doctor_name (str, optional)
      - clinic_name (str, optional)
    """
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    if not thread_id:
        return jsonify({ 'error': 'thread_id is required' }), 400

    # Build configurable parameters
    cfg = { 'thread_id': thread_id }
    if data.get('doctor_name'):
        cfg['doctor_name'] = data['doctor_name']
    if data.get('clinic_name'):
        cfg['clinic_name'] = data['clinic_name']

    # Initialize conversation state
    conversations[thread_id] = {
        'app': lance_main.app,
        'last_activity': datetime.utcnow(),
        'configurable': cfg
    }
    return jsonify({ 'message': f'Conversation {thread_id} started.' }), 200

@app.route('/message', methods=['POST'])
def send_message():
    """
    Send a user message to the chatbot.
    Request JSON should include:
      - thread_id (str)
      - message (str)
      - prescription (str, optional for followup bot)
    """
    data = request.get_json() or {}
    thread_id = data.get('thread_id')
    user_message = data.get('message')

    if not thread_id or not user_message:
        return jsonify({ 'error': 'thread_id and message are required' }), 400

    # Auto-start if missing
    if thread_id not in conversations:
        # For safety, require explicit start
        return jsonify({ 'error': 'Conversation not found. Call /start_conversation first.' }), 404

    # Check session timeout
    conv = conversations[thread_id]
    if datetime.utcnow() - conv['last_activity'] > SESSION_TIMEOUT:
        conversations.pop(thread_id, None)
        return jsonify({ 'error': 'Session expired after 15 minutes of inactivity.' }), 440

    # Update last activity
    conv['last_activity'] = datetime.utcnow()

    # Merge in prescription if provided
    configurable = conv['configurable'].copy()
    if data.get('prescription'):
        configurable['prescription'] = data['prescription']

    # Prepare LangGraph invocation
    config = { 'configurable': configurable }
    input_data = { 'messages': [HumanMessage(content=user_message)] }

    # Process in thread pool (parallel)
    future = executor.submit(lambda: conv['app'].invoke(input_data, config))
    state = future.result()

    # Extract last AI message
    reply = ''
    if state and state.get('messages'):
        for msg in reversed(state['messages']):
            if isinstance(msg, AIMessage):
                reply = msg.content
                break

    return jsonify({ 'reply': reply }), 200

if __name__ == '__main__':
    # Launch with threading enabled
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
