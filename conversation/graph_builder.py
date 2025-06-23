# conversation/graph_builder.py (Revised)
from langgraph.graph import StateGraph, END, START # START might not be strictly needed here anymore, but no harm in keeping it for now
from langgraph.checkpoint.memory import MemorySaver
from conversation.chat_state import ChatState
from conversation.nodes import get_info_node, symptom_node, followup_node # No need for same_episode_check_node, process_episode_response_node here as they are only used in the main graph if it existed
# from conversation.router import decide_bot_route # No need to import router here as it's not used in individual graph builders

def build_get_info_graph():
    """Builds and compiles the graph for the Get Info bot."""
    get_info_workflow = StateGraph(ChatState)
    get_info_workflow.add_node("get_info", get_info_node)
    get_info_workflow.set_entry_point("get_info")
    get_info_workflow.add_edge("get_info", END)
    return get_info_workflow.compile(checkpointer=MemorySaver())

def build_symptom_graph():
    """Builds and compiles the graph for the Symptom Collector bot."""
    symptom_workflow = StateGraph(ChatState)
    symptom_workflow.add_node("symptom_node", symptom_node)
    symptom_workflow.set_entry_point("symptom_node")
    symptom_workflow.add_edge("symptom_node", END)
    return symptom_workflow.compile(checkpointer=MemorySaver())

def build_followup_graph():
    """Builds and compiles the graph for the Follow-Up bot."""
    followup_workflow = StateGraph(ChatState)
    followup_workflow.add_node("followup", followup_node)
    followup_workflow.set_entry_point("followup")
    followup_workflow.add_edge("followup", END)
    return followup_workflow.compile(checkpointer=MemorySaver())

# Remove build_main_graph() from this file.