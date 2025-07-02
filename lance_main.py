# lance_main.py (Revised)

# Import necessary modules from the new structure
from config.llm_config import llm
from config.constants import CLINIC_INFO, CLINIC_CONFIG, SAMPLE_PRESCRIPTION
from conversation.graph_builder import build_get_info_graph, build_symptom_graph, build_followup_graph
from conversation.router import decide_bot_route
from conversation.chat_state import ChatState

# ====================
# LangGraph Setup (Individual Bots)
# ====================

# Build individual bot graphs
get_info_app = build_get_info_graph()
symptom_app = build_symptom_graph()
followup_app = build_followup_graph()

# Expose for app.py
__all__ = [
    "get_info_app",
    "symptom_app",
    "followup_app",
    "decide_bot_route",
    "ChatState",
    "CLINIC_INFO",
    "CLINIC_CONFIG",
    "SAMPLE_PRESCRIPTION"
]

print("Individual Bot Graphs compiled.")