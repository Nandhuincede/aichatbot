import re
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from state import ChatState
from llm import _base_llm, get_reply_llm
from instructions import OUT_OF_CONTEXT_DETECTOR, OUT_OF_CONTEXT_REPLY

logger = logging.getLogger(__name__)

QUESTION_WORDS = {"what", "when", "who", "which", "where", "why", "how", "whose", "whom"}

CONTEXT_CHECK_PROMPT = (
    "You are a strict guard for a contact-collection chatbot. "
    "The chatbot ONLY accepts: a person's real name, a phone number, or an email address. "
    "ANYTHING else is irrelevant — including questions, greetings, general knowledge, "
    "AI-related questions, opinions, or anything that is not a name/phone/email. "
    "Examples of IRRELEVANT: 'what is your name', 'how are you', 'what is AI', "
    "'tell me a joke', 'who are you', 'what can you do'. "
    "Examples of RELEVANT: 'John Smith', '9876543210', 'john@gmail.com', 'my name is Priya'. "
    "Reply with ONLY one word: 'relevant' or 'irrelevant'. No explanation."
)


def _is_data_attempt(user_input: str) -> bool:
    s = user_input.strip().lower()
    if any(c.isdigit() for c in s):
        return True
    if "@" in s:
        return True
    words = s.split()
    if len(words) <= 3 and all(w.isalpha() for w in words):
        return True
    return False


def _is_clear_question(user_input: str) -> bool:
    lowered = user_input.strip().lower()
    first_word = re.split(r'[\s?!,.]', lowered)[0]
    if first_word in QUESTION_WORDS:
        return True
    if lowered.endswith("?") and not any(c.isdigit() or c == "@" for c in lowered):
        return True
    return False


def is_out_of_context(user_input: str) -> bool:
    if _is_data_attempt(user_input):
        logger.info(f"[OOC CHECK] Skipped — looks like data attempt: '{user_input}'")
        return False
    if _is_clear_question(user_input):
        logger.info(f"[OOC CHECK] Caught by question filter: '{user_input}'")
        return True
    try:
        llm = _base_llm()
        response = llm.invoke([
            SystemMessage(content=CONTEXT_CHECK_PROMPT),
            HumanMessage(content=user_input)
        ])
        verdict = response.content.strip().lower()
        logger.info(f"[OOC CHECK] LLM verdict for '{user_input}': '{verdict}'")
        return "irrelevant" in verdict
    except Exception as e:
        logger.warning(f"[OOC CHECK] LLM call failed, defaulting to relevant: {e}")
        return False


def out_of_context_node(state: ChatState) -> ChatState:
    step = state.get("step", "collect_name")

    step_label = {
        "collect_name": "your full name",
        "collect_phone": "your mobile number",
        "collect_email": "your email address",
        "collect_description": "any additional message",
    }.get(step, "your details")

    # Step 4: LLM generates the OOC reply
    try:
        llm = get_reply_llm()
        prompt = OUT_OF_CONTEXT_REPLY.replace("{step_label}", step_label)
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Generate the reply for step: {step_label}")
        ])
        msg = response.content.strip()
        logger.info(f"[OOC NODE] LLM generated reply for step='{step}'")
    except Exception as e:
        logger.warning(f"[OOC NODE] LLM failed, using fallback: {e}")
        msg = (
            "I'm an AI chatbot built for collecting contact details. "
            f"I can't provide responses to anything other than that. "
            f"Could you please share {step_label}?"
        )

    return {
        **state,
        "last_bot_message": msg,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }
