import re
import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import ChatState, ExtractedName, NameValidation
from llm import get_extraction_llm, get_tool_calling_llm, get_structured_llm
from tools import validate_name_tool
from instructions import (
    NAME_EXTRACTOR, NAME_TOOL_CALLER, NAME_TOOL_CALLER_STRICT, NAME_VERDICT_INTERPRETER,
)

logger = logging.getLogger(__name__)

_NAME_RESERVED_WORDS = {
    "hello", "hi", "hey", "hiya", "howdy", "greetings", "sup", "yo",
    "yes", "no", "ok", "okay", "sure", "thanks", "thank", "bye",
    "goodbye", "start", "help", "test", "none", "null", "na", "n/a",
    "fullname", "yourname", "username", "name", "firstname", "lastname",
}


def _is_refusal(user_input: str) -> bool:
    refusal_patterns = [
        r'^(no|nope|nah|na|noo|nooo)$',
        r'^(i\s+)?(don\'t|dont|do not|wont|won\'t|will not|not going to|not gonna)\s+(want|give|share|tell|provide)',
        r'^(not\s+)?(interested|willing|sharing|telling|giving)',
        r'^(i\s+)?(refuse|decline|skip\s+this|pass)',
        r'^(why|why\s+should|why\s+do)',
        r'^(private|personal|confidential|none\s+of\s+your)',
        r"^(i\s+)?(don't|dont)\s+(have|want\s+to)",
        r'^no\s+(i|way|thanks|thank)',
    ]
    s = user_input.strip().lower()
    for p in refusal_patterns:
        if re.search(p, s, re.IGNORECASE):
            return True
    return False


def _refusal_msg(field: str) -> str:
    return (
        f"If you're unwilling to share your **{field}**, you're welcome to close the conversation. "
        f"If you'd like to continue, please go ahead and provide your **{field}**."
    )


def _regex_extract_name(user_input: str) -> str:
    patterns = [
        r"(?:my name is|i am|i'm|call me|name['\\s]?s?(?:\\s+is)?)\\s+([A-Za-z\\s]+)",
        r"^([A-Za-z]+(?:\\s[A-Za-z]+)*)$"
    ]
    for p in patterns:
        m = re.search(p, user_input.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return user_input.strip()


def run_name_validation(user_input: str) -> NameValidation:
    raw = user_input.strip()
    if re.search(r'\d', raw):
        return NameValidation(
            is_valid=False,
            reason="Name must not contain numbers. Please enter your real full name.",
            cleaned_name=""
        )

    if raw.lower() in _NAME_RESERVED_WORDS:
        return NameValidation(
            is_valid=False,
            reason=f"'{raw}' is a greeting, not a name. Please enter your real full name.",
            cleaned_name=""
        )

    # Step 1: Extract name via structured output
    try:
        extractor = get_extraction_llm(ExtractedName)
        extraction: ExtractedName = extractor.invoke([
            SystemMessage(content=NAME_EXTRACTOR),
            HumanMessage(content=f"Extract the name from: '{user_input}'")
        ])
        extracted = extraction.extracted_name.strip()
        logger.info(f"[NAME STEP 1] Extracted: '{user_input}' -> '{extracted}' [{extraction.confidence}]")
    except Exception as e:
        logger.warning(f"[NAME STEP 1] Structured extraction failed, using regex fallback: {e}")
        extracted = _regex_extract_name(user_input)

    # Step 2: LLM calls validate_name_tool
    tool_result_json = None
    try:
        tool_llm = get_tool_calling_llm(tools=[validate_name_tool])
        messages_step2 = [
            SystemMessage(content=NAME_TOOL_CALLER),
            HumanMessage(content=f"Validate this name using the tool: '{extracted}'")
        ]
        tool_response = tool_llm.invoke(messages_step2)
        tool_calls = getattr(tool_response, 'tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'validate_name_tool':
                    tool_result_json = validate_name_tool.invoke({'name': extracted})
                    logger.info(f"[NAME STEP 2] Tool called: validate_name_tool('{extracted}') -> {tool_result_json}")
        else:
            logger.warning(f"[NAME STEP 2] LLM skipped tool call. Retrying.")
            retry_response = tool_llm.invoke([
                SystemMessage(content=NAME_TOOL_CALLER_STRICT),
                HumanMessage(content=f"Call validate_name_tool now with name='{extracted}'")
            ])
            retry_calls = getattr(retry_response, 'tool_calls', [])
            if retry_calls:
                for tc in retry_calls:
                    if tc['name'] == 'validate_name_tool':
                        tool_result_json = validate_name_tool.invoke({'name': extracted})
                        logger.info(f"[NAME STEP 2] Retry succeeded: {tool_result_json}")
            else:
                logger.warning(f"[NAME STEP 2] Retry also skipped. Falling back to direct invocation.")
    except Exception as e:
        logger.warning(f"[NAME STEP 2] Tool calling failed: {e}")

    if not tool_result_json:
        tool_result_json = validate_name_tool.invoke({'name': extracted})
        logger.info(f"[NAME STEP 2] Direct fallback: {tool_result_json}")

    # Step 3: Interpret tool result via structured output
    try:
        structured_llm = get_structured_llm(NameValidation)
        result: NameValidation = structured_llm.invoke([
            SystemMessage(content=NAME_VERDICT_INTERPRETER),
            HumanMessage(content=(
                f"Name being validated: '{extracted}'\n"
                f"Tool result: {tool_result_json}\n"
                "Return the structured validation result."
            ))
        ])
        if result.is_valid and not result.cleaned_name:
            result.cleaned_name = extracted.title()
        logger.info(f"[NAME STEP 3] Verdict: valid={result.is_valid}, name='{result.cleaned_name}'")
        return result
    except Exception as e:
        logger.warning(f"[NAME STEP 3] Structured output failed, parsing directly: {e}")
        tool_data = json.loads(tool_result_json)
        return NameValidation(
            is_valid=tool_data['is_valid'],
            reason=tool_data['reason'],
            cleaned_name=extracted.title() if tool_data['is_valid'] else ""
        )


def collect_name_node(state: ChatState) -> ChatState:
    try:
        user_input = state.get("current_input", "")
        attempts = state.get("validation_attempts", 0)
        if _is_refusal(user_input):
            msg = _refusal_msg("full name")
            return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
        result: NameValidation = run_name_validation(user_input)
        
        if result.is_valid:
            name = result.cleaned_name or user_input.strip().title()

            next_step = "collect_phone" if not state.get("phone") else \
                "collect_email" if not state.get("email") else \
                "collect_description"

            next_msg = {
             "collect_phone": f"Great to meet you, **{name}**!\n\nCould you please share your **Indian mobile number**? (10 digits)",
             "collect_email": f"Great to meet you, **{name}**!\n\nCould you please share your **email address**?",
             "collect_description": f"Great to meet you, **{name}**!\n\nDo you have any **additional message or description**? (Type 'skip' to skip)",
            }.get(next_step)

            return {
             **state,
             "name": name,
             "step": next_step,
             "validation_attempts": 0,
             "last_bot_message": next_msg,
             "messages": state["messages"] + [AIMessage(content=next_msg)]
            }
        else:
            attempts += 1
            msg = f"Hmm, that doesn't look like a valid name — {result.reason}. Could you please enter your **full name**?"
            return {**state, "validation_attempts": attempts, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
    except Exception as e:
        logger.error(f"Error in collect_name_node: {e}")
        msg = "Sorry, I had a small hiccup. Could you please share your **full name** again?"
        return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
