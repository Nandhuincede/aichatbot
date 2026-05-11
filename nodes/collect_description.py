import re
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import ChatState
from llm import get_reply_llm
from instructions import (
    DESCRIPTION_REPLY_CLOSING,
    DESCRIPTION_REPLY_ANNOUNCE,
    DESCRIPTION_REPLY_QUESTION,
)

logger = logging.getLogger(__name__)

QUESTION_WORDS = {"what", "when", "who", "which", "where", "why", "how", "whose", "whom"}

ANNOUNCEMENT_PATTERNS = [
    r"^yes\b", r"^yeah\b", r"^sure\b", r"^ok\b", r"^okay\b",
    r"^i have\b", r"^i do have\b", r"^i want to\b",
    r"^i would like to\b", r"^i'd like to\b",
    r"^i have some\b", r"^yes i have\b", r"^yes,\s*i\b",
]

SKIP_WORDS = {"skip", "no", "none", "n/a", "na", "", "nope", "nah", "no thanks", "not now"}


def _is_announcement(text: str) -> bool:
    lowered = text.strip().lower()
    for pattern in ANNOUNCEMENT_PATTERNS:
        if re.match(pattern, lowered):
            return True
    return False


def _is_question(text: str) -> bool:
    lowered = text.strip().lower()
    first_word = re.split(r'[\s?!,.]', lowered)[0]
    if first_word in QUESTION_WORDS:
        return True
    if lowered.endswith("?") and not any(c.isdigit() or c == "@" for c in lowered):
        return True
    return False


def _generate_reply(prompt: str, context: str) -> str | None:
    try:
        llm = get_reply_llm()
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=context)
        ])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[DESCRIPTION REPLY] LLM failed: {e}")
        return None


def collect_description_node(state: ChatState) -> ChatState:
    user_input = state.get("current_input", "").strip()
    name = state.get("name", "there")

    # User is skipping
    if user_input.lower() in SKIP_WORDS:
        details = (
            f"Name: {state.get('name')}, "
            f"Phone: {state.get('phone')}, "
            f"Email: {state.get('email')}, "
            f"Description: none provided"
        )
        msg = _generate_reply(
            DESCRIPTION_REPLY_CLOSING,
            f"User's name: {name}. Collected details: {details}."
        ) or (
            f"Thank you, {name}! Here's a summary:\n"
            f"- Name: {state.get('name')}\n"
            f"- Phone: {state.get('phone')}\n"
            f"- Email: {state.get('email')}\n"
            f"- Description: —\n"
            f"We'll be in touch soon!"
        )
        return {
            **state,
            "description": None,
            "step": "done",
            "last_bot_message": msg,
            "messages": state["messages"] + [AIMessage(content=msg)]
        }

    # User asked a question
    if _is_question(user_input):
        msg = _generate_reply(
            DESCRIPTION_REPLY_QUESTION,
            f"User asked: '{user_input}' instead of providing a description."
        ) or (
            "I'm here only to collect contact details. "
            "Please share your additional message or type 'skip' to skip."
        )
        return {
            **state,
            "last_bot_message": msg,
            "messages": state["messages"] + [AIMessage(content=msg)]
        }

    # User announced they have a message but didn't provide it
    if _is_announcement(user_input):
        msg = _generate_reply(
            DESCRIPTION_REPLY_ANNOUNCE,
            f"User said: '{user_input}' — they announced a message but didn't write it yet."
        ) or "Ok! Please go ahead and provide your description."
        return {
            **state,
            "last_bot_message": msg,
            "messages": state["messages"] + [AIMessage(content=msg)]
        }

    # User provided actual description
    details = (
        f"Name: {state.get('name')}, "
        f"Phone: {state.get('phone')}, "
        f"Email: {state.get('email')}, "
        f"Description: {user_input}"
    )
    msg = _generate_reply(
        DESCRIPTION_REPLY_CLOSING,
        f"User's name: {name}. Collected details: {details}."
    ) or (
        f"Thank you, {name}! Here's a summary:\n"
        f"- Name: {state.get('name')}\n"
        f"- Phone: {state.get('phone')}\n"
        f"- Email: {state.get('email')}\n"
        f"- Description: {user_input}\n"
        f"We'll be in touch soon!"
    )
    return {
        **state,
        "description": user_input,
        "step": "done",
        "last_bot_message": msg,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }
