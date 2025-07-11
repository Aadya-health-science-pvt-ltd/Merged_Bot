# conversation/graph_builder.py (Revised)
import os
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.redis import RedisSaver
from conversation.chat_state import ChatState
from conversation.nodes import get_info_node, symptom_node, followup_node

# Redis configuration for persistent storage
REDIS_URL = os.getenv("REDIS_URL")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "chatbot:state:")

def build_get_info_graph():
    """Builds and compiles the graph for the Get Info bot."""
    get_info_workflow = StateGraph(ChatState)
    get_info_workflow.add_node("get_info", get_info_node)
    get_info_workflow.set_entry_point("get_info")
    get_info_workflow.add_edge("get_info", END)
    
    # Use Redis for persistent storage
    with RedisSaver.from_conn_string(REDIS_URL, key_prefix=REDIS_KEY_PREFIX) as checkpointer:
        checkpointer.setup()  # Initialize Redis indices
        return get_info_workflow.compile(checkpointer=checkpointer)

def build_symptom_graph():
    """Builds and compiles the graph for the Symptom Collector bot."""
    symptom_workflow = StateGraph(ChatState)
    symptom_workflow.add_node("symptom", symptom_node)
    symptom_workflow.set_entry_point("symptom")
    symptom_workflow.add_edge("symptom", END)
    
    # Use Redis for persistent storage
    with RedisSaver.from_conn_string(REDIS_URL, key_prefix=REDIS_KEY_PREFIX) as checkpointer:
        checkpointer.setup()  # Initialize Redis indices
        return symptom_workflow.compile(checkpointer=checkpointer)

def build_followup_graph():
    """Builds and compiles the graph for the Follow-Up bot."""
    followup_workflow = StateGraph(ChatState)
    followup_workflow.add_node("followup", followup_node)
    followup_workflow.set_entry_point("followup")
    followup_workflow.add_edge("followup", END)
    
    # Use Redis for persistent storage
    with RedisSaver.from_conn_string(REDIS_URL, key_prefix=REDIS_KEY_PREFIX) as checkpointer:
        checkpointer.setup()  # Initialize Redis indices
        return followup_workflow.compile(checkpointer=checkpointer)

def debug_print_thread_state(graph, thread_id):
    """Prints the latest state and full state history for a given thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    # Print latest state
    latest_state = graph.get_state(config)
    print(f"\n[DEBUG] Latest state for thread {thread_id}:")
    print(latest_state)
    # Print full state history
    state_history = list(graph.get_state_history(config))
    print(f"\n[DEBUG] State history for thread {thread_id}:")
    for i, snapshot in enumerate(state_history):
        print(f"Checkpoint {i}: {snapshot}\n")

# Remove build_main_graph() from this file.