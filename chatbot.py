"""
chatbot.py — Public API for the chatbot.

Graph construction  → graph.py
LLM factories       → llm.py
Node logic          → nodes/
State & schemas     → state.py
Validation tools    → tools.py
"""

import logging
from langchain_core.messages import HumanMessage

from graph import chatbot_graph
from state import ChatState

logger = logging.getLogger(__name__)


def process_message(session_id: str, session_state: dict, user_message: str = None) -> tuple:
    """
    Routes through the compiled LangGraph graph using MemorySaver checkpointing.

    Turn 1 (greeting): session_state has no 'step' yet — we initialise the full
    ChatState and invoke. MemorySaver writes the checkpoint.

    Turn 2+ (collect_*): MemorySaver already holds the full state for this
    thread_id. We pass ONLY the fields that change each turn (current_input,
    last_bot_message reset, and the new HumanMessage). LangGraph merges these
    with the saved checkpoint via the add_messages reducer — the step, name,
    phone, email fields are restored automatically from the checkpoint.

    Returns: (bot_response: str, lightweight_state: dict)
    """
    config = {"configurable": {"thread_id": session_id}}

    if not session_state.get("step"):
        # First turn: full initialisation
        state: ChatState = {
            "messages": [],
            "session_id": session_id,
            "step": "greeting",
            "name": None,
            "phone": None,
            "email": None,
            "description": None,
            "current_input": None,
            "validation_attempts": 0,
            "last_bot_message": None,
        }
    else:
        # Subsequent turns: pass only the delta
        state: ChatState = {
            "messages": [HumanMessage(content=user_message or "")],
            "current_input": user_message,
            "last_bot_message": None,
            "session_id": session_id,
            "step": session_state.get("step", "greeting"),
            "name": session_state.get("name"),
            "phone": session_state.get("phone"),
            "email": session_state.get("email"),
            "description": session_state.get("description"),
            "validation_attempts": session_state.get("validation_attempts", 0),
        }

    result = chatbot_graph.invoke(state, config=config)

    # Return bot reply + a lightweight dict (no messages list) for Flask session_store
    lightweight = {
        k: v for k, v in result.items()
        if k not in ("messages",)
    }
    return result["last_bot_message"], lightweight
